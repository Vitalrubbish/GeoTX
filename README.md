# GeoTX: A Transfer Learning Example of Geo-Locating on StreetView Dataset

## Introduction

This is the course project of Machine Learning (CS3308), based on [GeoCLIP](https://arxiv.org/abs/2309.16020v2) (NeurIPS 2023). For documents of GeoCLIP, see `docs/geoclip`.

Traditional geo-localization models like GeoCLIP achieve high accuracy at thresholds of 1km, 25km, 200km, 750km, and 2500km. However, their training dataset (MP-16) and testing dataset (Im2GPS3k) are dominated by tourist images — landmarks, distinctive architecture, iconic scenery. These images contain strong visual cues that make location recognition relatively straightforward.

Street-view images present a much harder problem: repetitive building facades, generic roads, fewer landmarks, and highly variable lighting/weather. Consequently, GeoCLIP's accuracy degrades significantly on street-view datasets.

**GeoTX** is a transfer-learning pipeline that progressively adapts GeoCLIP to street-view images through three targeted optimizations:

| Version | Components | Key Innovation |
|---------|-----------|----------------|
| **GeoTX v0.1** | SigmaSelector + unfrozen capsule heads | Learnable location-conditioned branch fusion |
| **GeoTX v0.2** | v0.1 + LoRA on CLIP ViT + image MLP | Domain-adaptive visual features via low-rank adaptation |
| **GeoTX v0.3** | v0.2 + Optimized Negative Sampling | Geography-aware contrastive loss |

## Architecture

### Overall Pipeline

```mermaid
graph TB
    subgraph Input
        IMG["Street-View Image<br/>224×224×3"]
        GPS["GPS Coordinate<br/>(lat, lon)"]
    end

    subgraph ImageEncoder ["Image Encoder"]
        VIT["CLIP ViT-L/14<br/>Frozen Backbone"]
        LORA["LoRA Adapters<br/>(q_proj, v_proj)<br/>Layers 18-23"]
        MLP["Image MLP Head<br/>768 → 768 → 512"]
        VIT --> LORA
        LORA --> MLP
        IMG --> VIT
    end

    subgraph LocationEncoder ["Location Encoder"]
        EE["Equal Earth<br/>Projection"]
        RFF0["LocEnc0<br/>RFF σ=1"]
        RFF1["LocEnc1<br/>RFF σ=16"]
        RFF2["LocEnc2<br/>RFF σ=256"]
        SS["SigmaSelector<br/>GPS → weights"]
        FUSION["Weighted Sum<br/>Σ w_i · f_i"]

        GPS --> EE
        EE --> RFF0
        EE --> RFF1
        EE --> RFF2
        EE --> SS
        RFF0 --> FUSION
        RFF1 --> FUSION
        RFF2 --> FUSION
        SS --> FUSION
    end

    subgraph Contrastive ["Contrastive Learning"]
        I_FEAT["Image Features<br/>(B, 512)"]
        L_FEAT["Location Features<br/>(B, 512)"]
        SIM["Cosine Similarity<br/>× logit_scale"]
        QUEUE["GPS Queue<br/>4096 stored embeddings"]
        MASK["Negative Sample Mask<br/>Exclude close negatives"]
        LOSS["Cross-Entropy Loss"]

        MLP --> I_FEAT
        FUSION --> L_FEAT
        I_FEAT --> SIM
        L_FEAT --> SIM
        QUEUE --> SIM
        SIM --> MASK
        MASK --> LOSS
    end

    style LORA fill:#e74c3c,color:#fff
    style SS fill:#3498db,color:#fff
    style MASK fill:#2ecc71,color:#fff
    style FUSION fill:#3498db,color:#fff
```

**Legend:** Red = LoRA (v0.2), Blue = SigmaSelector (v0.1), Green = Negative Sampling (v0.3)

### Location Encoder Detail: Three-Branch RFF + SigmaSelector

GeoCLIP's LocationEncoder models the Earth's surface as a continuous function using Random Fourier Features (RFF). GPS coordinates are first projected via Equal Earth projection, then encoded through three parallel branches with different spatial-frequency sensitivities (σ = 1, 16, 256):

```mermaid
graph LR
    subgraph EqualEarthProjection ["Equal Earth Projection"]
        GPS2["(lat, lon)"] --> EE2["EE(x, y)"]
    end

    subgraph Branch0 ["Branch 0: σ=1 (Fine)"]
        EE2 --> R0["GaussianEncoding<br/>size=256"] --> C0["Capsule<br/>512→1024→1024→1024"] --> H0["Head<br/>1024→512"]
    end

    subgraph Branch1 ["Branch 1: σ=16 (Medium)"]
        EE2 --> R1["GaussianEncoding<br/>size=256"] --> C1["Capsule<br/>512→1024→1024→1024"] --> H1["Head<br/>1024→512"]
    end

    subgraph Branch2 ["Branch 2: σ=256 (Coarse)"]
        EE2 --> R2["GaussianEncoding<br/>size=256"] --> C2["Capsule<br/>512→1024→1024→1024"] --> H2["Head<br/>1024→512"]
    end

    subgraph SigmaSelector ["SigmaSelector"]
        EE2 --> SS2["Linear(2→64)<br/>ReLU<br/>Linear(64→3)<br/>Softmax"] --> W["w₀, w₁, w₂"]
    end

    subgraph Fusion ["Fusion"]
        H0 --> SUM["Σ = w₀·f₀ + w₁·f₁ + w₂·f₂"]
        H1 --> SUM
        H2 --> SUM
        W --> SUM
        SUM --> OUT["Location Embedding<br/>(B, 512)"]
    end

    style SS2 fill:#3498db,color:#fff
    style W fill:#3498db,color:#fff
```

**Key insight:** The original GeoCLIP sums the three branch outputs equally. SigmaSelector replaces this with a learnable, GPS-conditioned weighted sum, letting the model learn which spatial scale matters most for each geographic region.

### Three-Stage Optimization Evolution

```mermaid
graph LR
    subgraph GeoCLIPBaseline ["GeoCLIP Baseline"]
        B_IMG["CLIP ViT<br/>(fully frozen)"] --> B_MLP["MLP<br/>(frozen)"]
        B_LOC["3-Branch RFF<br/>Equal-weight sum<br/>(frozen)"] 
    end

    subgraph GeoTXv01 ["GeoTX v0.1"]
        V1_IMG["CLIP ViT<br/>(frozen)"] --> V1_MLP["MLP<br/>(frozen)"]
        V1_LOC["3-Branch RFF<br/>(capsule heads unfrozen)"] 
        V1_SS["SigmaSelector<br/>(trainable)"] --> V1_F["Weighted Sum"]
        V1_LOC --> V1_F
    end

    subgraph GeoTXv02 ["GeoTX v0.2"]
        V2_IMG["CLIP ViT<br/>+ LoRA (trainable)"] --> V2_MLP["MLP<br/>(trainable)"]
        V2_LOC["3-Branch RFF<br/>(capsule heads unfrozen)"]
        V2_SS["SigmaSelector<br/>(trainable)"] --> V2_F["Weighted Sum"]
        V2_LOC --> V2_F
    end

    subgraph GeoTXv03 ["GeoTX v0.3"]
        V3_IMG["CLIP ViT<br/>+ LoRA (trainable)"] --> V3_MLP["MLP<br/>(trainable)"]
        V3_LOC["3-Branch RFF<br/>(capsule heads unfrozen)"]
        V3_SS["SigmaSelector<br/>(trainable)"] --> V3_F["Weighted Sum"]
        V3_LOC --> V3_F
        V3_NEG["Geographic Negative<br/>Sample Mask"] -.-> V3_LOSS["Contrastive Loss"]
    end

    B_IMG -.-> V1_IMG
    V1_IMG -.-> V2_IMG
    V2_IMG -.-> V3_IMG

    style V1_SS fill:#3498db,color:#fff
    style V1_F fill:#3498db,color:#fff
    style V2_IMG fill:#e74c3c,color:#fff
    style V2_MLP fill:#e74c3c,color:#fff
    style V3_NEG fill:#2ecc71,color:#fff
    style V3_LOSS fill:#2ecc71,color:#fff
```

## Testing

See `docs/dev/test.md` for the complete training and evaluation runbook for all three versions.

## Project Structure

```
geo-clip/
├── geoclip/
│   ├── model/
│   │   ├── GeoCLIP.py           # Main model + haversine + negative sampling mask
│   │   ├── image_encoder.py     # CLIP ViT + LoRA + MLP head
│   │   ├── location_encoder.py  # 3-branch RFF + SigmaSelector weighted fusion
│   │   ├── misc.py              # GPS data loading utilities
│   │   └── rff/                 # Random Fourier Features (Gaussian encoding)
│   └── train/
│       ├── train.py             # Original GeoCLIP training loop
│       ├── dataloader.py        # GeoDataLoader + image transforms
│       └── eval.py              # Distance-accuracy evaluation
├── scripts/
│   ├── train_sigma_selector.py      # Train GeoTX v0.1
│   ├── eval_sigma_selector.py       # Evaluate v0.1 vs baseline
│   ├── train_lora.py                # Train GeoTX v0.2
│   ├── eval_lora.py                 # Evaluate v0.2/v0.3
│   ├── train_negative_sampling.py   # Train GeoTX v0.3
│   ├── sample_streetview_subset.py  # Feasibility subset (900/100 split)
│   └── convert_streetview_pano_dataset.py
└── docs/
    ├── dev/              # Design specs for each optimization
    ├── geoclip/          # Original GeoCLIP docs
    └── reproduction/     # Im2GPS3k reproduction logs
```

## Evaluation Metrics

All models are evaluated with Great-Circle Distance accuracy at five thresholds:

| Threshold | What it measures |
|-----------|-----------------|
| 1 km | City-block level precision |
| 25 km | City-scale localization |
| 200 km | Regional recognition |
| 750 km | Country/state level |
| 2500 km | Continental level |

## Acknowledgments

This project builds on [GeoCLIP](https://github.com/VicenteVivan/geo-clip) (NeurIPS 2023) by Vicente Vivanco, Gaurav Kumar Nayak, and Mubarak Shah. The Random Fourier Features implementation originates from Joshua M. Long's [random-fourier-features-pytorch](https://github.com/jmclong/random-fourier-features-pytorch).
