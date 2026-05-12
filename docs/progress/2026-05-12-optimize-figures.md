# Goal
Optimize the layout of the generated figures by alleviating text overlays, misaligned flow arrows, and squeezed bounding boxes in all the visual charts under the `figures` folder. 

# Changes
- Completely refactored `scripts/generate_figures.py` layout bounds (margins, padding, box width).
- Solved clipping arrows against box text. 
- Alleviated text overlapping throughout image/spatial components by widening bounding boxes and resizing tight column grid splits.

## Location Encoder specific fixes
- Extracted overlapping diagonal flow arrows crossing the $\sigma$ variant boxes into cleanly routed orthogonal branch lines.
- The input splits now correctly form a main "trunk" beside the boxes, pointing horizontally into each $\sigma$ layer and channeling down out to the Fusion unit, preventing any occlusion on the labels.
