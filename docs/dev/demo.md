# Demo for GeoTX: Adapting GeoCLIP for Streetview Geolocation

## Introduction
Following the integration of **SigmaSelector** (Adaptive Frequency Selection), **LoRA** (Low-Rank Adaptation for parameter-efficient fine-tuning), and **Negative Sampling** into the GeoCLIP backbone, the development phase of **GeoTX** is successfully concluded. 

To intuitively interact with the model and generate compelling qualitative results for our mini-paper, we have designed a comprehensive suite of interactive demos using Jupyter Notebook.

## Environment & Materials Preparation
*   **Pre-trained Weights:** GeoTX LoRA weights and trained LocationEncoder weights.
*   **Base Map:** A high-resolution world map using the Mercator Projection (implemented via `Cartopy`, `Basemap`, or interactive `Folium`).
*   **Pre-computed Metadata:** A `.csv` file containing lat/lon pairs of the `streetview_pano` dataset (to prevent runtime overload).

---

## Demo 1: Interactive Geolocation Prediction (User-Facing)
**Objective:** Provide users with a straightforward, interactive experience of the GeoTX model, showcasing its global localization capabilities.

**Workflow in Jupyter Notebook:**
1.  **Loading Weights:** Initialize the GeoTX model and load the fine-tuned LoRA and LocationEncoder weights from `geoclip/model/weights`. *(Logs printed: "Model & Weights loaded successfully.")*
2.  **Receiving Input:** Utilize `ipywidgets.FileUpload` to allow users to directly upload an image from their local machine within the notebook. *(Logs printed: "Image received and preprocessed.")*
3.  **Evaluate & Visualize:** 
    *   Run the forward pass to get the global probability distribution.
    *   Extract the Top-1 predicted position (Longitude and Latitude).
    *   **Visualization:** Render a Mercator projection map. Plot the global **Probability Heatmap** to show model confidence, and place a highly visible marker (e.g., a Red Star) at the predicted location. *(Logs printed: "Evaluation complete. Prediction plotted.")*

---

## Demo 2: Spatial Distribution of Training Data
**Objective:** Visualize the geographic distribution of the `streetview_pano` fine-tuning dataset to provide context for the model's performance biases in the mini-paper.

**Workflow in Jupyter Notebook:**
1.  **Load Pre-computed Data:** Load the pre-extracted `dataset_coordinates.csv` containing all ground-truth (lat, lon) pairs from the dataset. *(Note: Avoid real-time dataset traversal to ensure notebook responsiveness).*
2.  **Generate Density Map:** Use 2D Kernel Density Estimation (KDE) or 2D Histograms to compute the spatial density.
3.  **Visualize:** Overlay the density data onto the Mercator world map as a **Hot-Cold Heatmap**. "Hot" areas (e.g., red/yellow) will indicate regions with a dense concentration of street-view data (e.g., US, Europe), while "Cold" areas (blue/transparent) indicate sparse regions.

---

## Demo 3: Interpretability of SigmaSelector
**Objective:** Analyze how the semantic content of a street-view image influences the routing weights of the LocationEncoder across different $\sigma$ (frequency scales). SigmaSelector is **image-conditioned** — it takes both image features and GPS coordinates as input, producing per-(image, GPS) routing weights.

**Hypothesis:** The model assigns higher weights to large $\sigma$ (high-frequency/fine details) for dense urban scenes, and lower $\sigma$ (low-frequency/global patterns) for natural landscapes like deserts or forests. Two different images at the same GPS coordinate should get different sigma routing.

**Workflow in Jupyter Notebook:**
1.  **Load Curated Examples:** Load a pre-selected gallery of 8-10 highly representative images from the dataset (e.g., 4 extreme urban cities, 4 extreme natural landscapes).
2.  **Extract Routing Weights:** Feed images and GPS coordinates into GeoTX and extract the per-pair attention weights produced by the image-conditioned SigmaSelector for each $\sigma$ layer.
3.  **Side-by-Side Visualization:** For each example, plot:
    *   *(Left)* The input street-view image.
    *   *(Right)* A Bar Chart showing the weight distribution across different $\sigma$ values, clearly demonstrating the model's adaptive scale selection based on visual content.

---

## Demo 4: Granularity and Accuracy Thresholds
**Objective:** Demonstrate GeoTX's performance across different geographic scales (City, State, Country, Continent) for the evaluation section of the mini-paper.

**Workflow in Jupyter Notebook:**
1.  **Select Threshold Cases:** Load pre-selected images that represent prediction errors within specific thresholds:
    *   **< 25 km** (City/Street level precision)
    *   **< 200 km** (State/Region level precision)
    *   **< 750 km** (Country level precision)
    *   **> 2500 km** (Continent level / Failure cases)
2.  **Comparative Visualization:** For each category, display a comprehensive result panel containing:
    *   The Input Image.
    *   A cropped map showing the **Ground Truth (Green Pin)**, the **Prediction (Red Pin)**, and a connecting line representing the error distance.
    *   The **Probability Heatmap** (This is crucial: for <25km, the heatmap should show a sharp, single peak; for >2500km, it will visually explain the model's confusion by showing a dispersed or multi-modal distribution).