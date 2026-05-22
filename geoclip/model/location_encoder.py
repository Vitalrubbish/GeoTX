import torch
import torch.nn as nn
import torch.nn.functional as F
from .rff import GaussianEncoding
from .misc import file_dir

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180

class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class SigmaSelector(nn.Module):
    """GPS-only SigmaSelector: routing weights depend only on geographic location."""

    def __init__(self, input_dim=2, num_sigmas=3, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sigmas),
            nn.Softmax(dim=-1),
        )

        # Start from uniform branch weights before training.
        nn.init.zeros_(self.attention[2].weight)
        nn.init.zeros_(self.attention[2].bias)

    def forward(self, location):
        return self.attention(location)


class ImageConditionedSigmaSelector(nn.Module):
    """v0.1: Image-conditioned SigmaSelector.

    Routing weights depend on BOTH image content and geographic location.
    Concatenates [image_features (512) | projected_gps (2)] = 514 dims
    and passes through an MLP to produce per-(image, GPS) routing weights.
    """

    def __init__(self, image_dim=512, gps_dim=2, num_sigmas=3, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        in_dim = image_dim + gps_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_sigmas))
        self.mlp = nn.Sequential(*layers)

        # Zero-init final layer for uniform starting weights
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, gps_proj, image_features):
        """Compute per-pair routing weights.

        Args:
            gps_proj (torch.Tensor): Equal-Earth projected GPS of shape (M, 2)
            image_features (torch.Tensor): Image embeddings of shape (N, D)

        Returns:
            torch.Tensor: Routing weights of shape (N, M, num_sigmas), softmax over last dim
        """
        N, D = image_features.shape
        M, G = gps_proj.shape

        img_exp = image_features.unsqueeze(1).expand(N, M, D)   # (N, M, D)
        gps_exp = gps_proj.unsqueeze(0).expand(N, M, G)          # (N, M, G)
        combined = torch.cat([img_exp, gps_exp], dim=-1)         # (N, M, D+G)

        logits = self.mlp(combined)                              # (N, M, num_sigmas)
        return F.softmax(logits, dim=-1)


class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=True,
                 use_sigma_selector=False, selector_variant=None):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.use_sigma_selector = use_sigma_selector
        self.selector_variant = selector_variant

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        if self.use_sigma_selector:
            if selector_variant == "v0.1":
                self.sigma_selector = ImageConditionedSigmaSelector(
                    image_dim=512, gps_dim=2, num_sigmas=self.n,
                )
            else:
                self.sigma_selector = SigmaSelector(input_dim=2, num_sigmas=self.n)

        if from_pretrained:
            self._load_weights()

    def _load_weights(self):
        state_dict = torch.load(f"{file_dir}/weights/location_encoder_weights.pth")
        self.load_state_dict(state_dict, strict=not self.use_sigma_selector)

    def get_sigma_weights(self, location, image_features=None):
        """Return SigmaSelector routing weights for interpretability.

        Args:
            location (torch.Tensor): GPS tensor of shape (M, 2)
            image_features (torch.Tensor, optional): Image embeddings of shape (N, D).
                Required when selector_variant="v0.1".

        Returns:
            torch.Tensor: Routing weights of shape (M, n) for GPS-only,
                          or (N, M, n) for image-conditioned.
        """
        if not self.use_sigma_selector:
            raise RuntimeError("SigmaSelector is not enabled. Set use_sigma_selector=True.")
        gps_proj = equal_earth_projection(location)
        if self.selector_variant == "v0.1":
            if image_features is None:
                raise ValueError("selector_variant='v0.1' requires image_features")
            return self.sigma_selector(gps_proj, image_features)
        else:
            return self.sigma_selector(gps_proj)

    def forward(self, location, image_features=None):
        gps_proj = equal_earth_projection(location)

        branch_features = []
        for i in range(self.n):
            branch_features.append(self._modules['LocEnc' + str(i)](gps_proj))
        stacked = torch.stack(branch_features, dim=1)  # (M, n, 512)

        if self.use_sigma_selector:
            if self.selector_variant == "v0.1":
                if image_features is None:
                    raise ValueError("selector_variant='v0.1' requires image_features")
                weights = self.n * self.sigma_selector(gps_proj, image_features)  # (N, M, n)
                weights = weights.unsqueeze(-1)          # (N, M, n, 1)
                stacked_exp = stacked.unsqueeze(0)       # (1, M, n, 512)
                location_features = (weights * stacked_exp).sum(dim=2)  # (N, M, 512)
            else:
                weights = self.n * self.sigma_selector(gps_proj).unsqueeze(-1)  # (M, n, 1)
                location_features = (weights * stacked).sum(dim=1)              # (M, 512)
        else:
            location_features = torch.zeros(location.shape[0], 512).to(location.device)
            for feature in branch_features:
                location_features += feature

        return location_features