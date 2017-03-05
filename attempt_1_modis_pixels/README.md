# Pixel-level Classification

This directory contains four attempts to classify Ethiopian wheat health on the per-field level.

Since our observations are at the per-field level, this approach seemed the simplest and most straightforward way to tackle the problem. Unfortunately, the spatial resolution of our MODIS imagery is 500m per pixel. This means that multiple farms may be swallowed by a single pixel in our training data. It is likely due to this problem that none of the following methods worked very well. 

## Methods

#### 1: Raw pixel classifier. 