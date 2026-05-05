# GeoTX: A Transfer Learning Example of Geo-Locating on StreetView Dataset
## Introduction
This is the course project of Machine Learning (CS3308), based on GeoCLIP. For documents of GeoCIP, you can move to `docs/geoclip`. 

This project is a transfer learning example. Traditional Geo-Locating models like geoclip (NeurIPS 2023) already has a high accuracy of locating images at the threshold of 1km, 25km, 200km, 750km and 2500km. However, its training dataset (MP16) and testing dataset (Im2gps3k) are mainly tourist images, which are often easy to to be recognized. Therefore, these models often have a relevantly lower performance on streetview datasets, where markable symbols are shown less frequently. 

GeoTX is a transfer learning example, aiming to enhance the ability (accuracy) of locating streetview images. 

## Development
The development and testing documents are in `docs/dev`.  You can reproduct the results by following instructions in `docs/dev/test.md`.

### SigmaSelector

### LoRA on ImageEncoder

### Negative Sampling (TBD)