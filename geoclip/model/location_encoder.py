import torch
import torch.nn as nn
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

class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=True, use_sigma_selector=False):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.use_sigma_selector = use_sigma_selector

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        if self.use_sigma_selector:
            self.sigma_selector = SigmaSelector(input_dim=2, num_sigmas=self.n)

        if from_pretrained:
            self._load_weights()

    def _load_weights(self):
        state_dict = torch.load(f"{file_dir}/weights/location_encoder_weights.pth")
        self.load_state_dict(state_dict, strict=not self.use_sigma_selector)

    def get_sigma_weights(self, location):
        """Return SigmaSelector routing weights for interpretability.

        Args:
            location (torch.Tensor): GPS tensor of shape (B, 2)

        Returns:
            torch.Tensor: Routing weights of shape (B, n)
        """
        if not self.use_sigma_selector:
            raise RuntimeError("SigmaSelector is not enabled. Set use_sigma_selector=True.")
        location = equal_earth_projection(location)
        return self.sigma_selector(location)

    def forward(self, location):
        location = equal_earth_projection(location)

        branch_features = []
        for i in range(self.n):
            branch_features.append(self._modules['LocEnc' + str(i)](location))

        if self.use_sigma_selector:
            weights = self.n * self.sigma_selector(location).unsqueeze(-1)  # (B, n, 1)
            stacked = torch.stack(branch_features, dim=1)  # (B, n, 512)
            location_features = (weights * stacked).sum(dim=1)
        else:
            location_features = torch.zeros(location.shape[0], 512).to(location.device)
            for feature in branch_features:
                location_features += feature

        return location_features