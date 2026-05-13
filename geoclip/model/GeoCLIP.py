import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage


def haversine_distance(gps_a, gps_b):
    """Compute pairwise haversine distance (km) between two sets of GPS coordinates.

    Args:
        gps_a (torch.Tensor): GPS tensor of shape (N, 2) in (lat, lon) degrees
        gps_b (torch.Tensor): GPS tensor of shape (M, 2) in (lat, lon) degrees

    Returns:
        torch.Tensor: Distance matrix of shape (N, M) in kilometers
    """
    R = 6371.0
    lat_a = torch.deg2rad(gps_a[:, 0:1])
    lon_a = torch.deg2rad(gps_a[:, 1:2])
    lat_b = torch.deg2rad(gps_b[:, 0:1])
    lon_b = torch.deg2rad(gps_b[:, 1:2])

    dlat = lat_a - lat_b.t()
    dlon = lon_a - lon_b.t()

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_a) * torch.cos(lat_b.t()) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c


def negative_sample_mask(gps_all, batch_size, neg_strategy, neg_threshold=200.0, neg_topk=None):
    """Build a boolean mask to exclude geographically close negatives.

    Args:
        gps_all (torch.Tensor): All GPS candidates of shape (N, 2) where N = batch_size + queue_size.
                                First batch_size entries are the positive anchors (and in-batch negatives).
        batch_size (int): Number of images/GPS in the current mini-batch.
        neg_strategy (str): "threshold" or "topk".
        neg_threshold (float): Distance threshold in km for "threshold" strategy.
        neg_topk (int | None): Number of furthest negatives to keep for "topk" strategy.

    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, N). True means "exclude this negative"
                      (set logit to -inf). Diagonal entries (positive pairs) are always False.
    """
    N = gps_all.shape[0]
    device = gps_all.device

    if neg_strategy == "threshold":
        distances = haversine_distance(gps_all[:batch_size], gps_all)
        mask = distances < neg_threshold
        mask[torch.arange(batch_size, device=device), torch.arange(batch_size, device=device)] = False

    elif neg_strategy == "topk":
        if neg_topk is None or neg_topk >= N - 1:
            return torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        distances = haversine_distance(gps_all[:batch_size], gps_all)
        mask = torch.ones(batch_size, N, dtype=torch.bool, device=device)
        mask[torch.arange(batch_size, device=device), torch.arange(batch_size, device=device)] = False

        sorted_indices = torch.argsort(distances, dim=1, descending=True)
        keep_indices = sorted_indices[:, :neg_topk]
        for i in range(batch_size):
            mask[i, keep_indices[i]] = False

    else:
        raise ValueError(f"Unknown neg_strategy: {neg_strategy}")

    return mask

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096, use_sigma_selector=False,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 selector_variant=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder(use_lora=use_lora, lora_r=lora_r,
                                          lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                          from_pretrained=from_pretrained)
        self.location_encoder = LocationEncoder(use_sigma_selector=use_sigma_selector,
                                                from_pretrained=from_pretrained,
                                                selector_variant=selector_variant)
        self.use_sigma_selector = use_sigma_selector
        self.use_lora = use_lora
        self.selector_variant = selector_variant

        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv"))
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(
            torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth")
        )
        self.location_encoder.load_state_dict(
            torch.load(f"{self.weights_folder}/location_encoder_weights.pth"),
            strict=not self.use_sigma_selector,
        )
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
                                             
    def forward(self, image, location, gallery_chunk_size=None):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)
            gallery_chunk_size (int | None): Chunk size for v0.1 selector to avoid OOM
                when m is large (e.g. 100K gallery).  If None, defaults to 4096.

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features = F.normalize(image_features, dim=1)

        if self.selector_variant == "v0.1":
            return self._forward_v01(image_features, location, logit_scale, gallery_chunk_size)
        else:
            location_features = self.location_encoder(location, image_features=image_features)
            location_features = F.normalize(location_features, dim=1)
            logits_per_image = logit_scale * (image_features @ location_features.t())
            return logits_per_image

    def _forward_v01(self, image_features, location, logit_scale, gallery_chunk_size):
        """Chunked forward for v0.1 selector to keep memory bounded when m is large."""
        m = location.shape[0]
        chunk_size = gallery_chunk_size or 4096

        if m <= chunk_size:
            location_features = self.location_encoder(location, image_features=image_features)
            location_features = F.normalize(location_features, dim=2)
            return logit_scale * torch.einsum('nd,nmd->nm', image_features, location_features)

        logit_chunks = []
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            loc_chunk = location[start:end]
            loc_feat_chunk = self.location_encoder(loc_chunk, image_features=image_features)
            loc_feat_chunk = F.normalize(loc_feat_chunk, dim=2)
            logit_chunk = logit_scale * torch.einsum('nd,nmd->nm', image_features, loc_feat_chunk)
            logit_chunks.append(logit_chunk)

        return torch.cat(logit_chunks, dim=1)

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob