# 2026-05-24: Academic Poster Generation

## Goal
The goal of this change was to generate an academic poster for the "GeoTX: Progressive Transfer Learning for Street-View Image Geo-Localization" project to facilitate presentations based on the contents written in the included mini-paper.

## Changes
- Created `minipaper/poster.tex` leveraging the `tikzposter` LaTeX document class.
- Extracted and summarized the core abstract, introduction, methodology (SigmaSelector, LoRA, and Geographic Negative Sampling), architecture pipeline figure, experimental results, and analyses into dedicated poster blocks.
- Formatted tables and included references to existing project figures (`figures/overall_pipeline_tikz.pdf` and `figures/sigma_urban_vs_natural.pdf`).

## Result
A fully formatted LaTeX poster document is now available at `minipaper/poster.tex`, which can be compiled directly via `pdflatex poster.tex` to produce a high-quality presentation poster.