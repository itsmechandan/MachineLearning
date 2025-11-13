# Machine Learning Classification of Gamma-Ray Events from Atmospheric Cherenkov Telescope Data

# Project Overview

CosmicClassifier is a machine learning project that classifies high-energy cosmic events recorded by an atmospheric Cherenkov telescope.
The goal is to distinguish gamma-ray events (signal) from hadronic cosmic-ray events (background) based on features derived from the MAGIC Gamma Telescope dataset.

These events are simulated using Monte Carlo techniques, and each observation is represented by 10 numerical parameters (known as Hillas parameters), which describe the geometric and brightness characteristics of the Cherenkov shower image.

In simple terms, the project teaches a model to tell apart true gamma-ray signals from noise or cosmic background particles, using real-world physics-inspired data.

# Dataset Information

Dataset Source: UCI Machine Learning Repository – MAGIC Gamma Telescope Dataset
Instances: 19,020 simulated events
Features: 10 continuous numerical variables (ellipse geometry, intensity ratios, etc.)

Target Variable:

g → Gamma-ray (signal)

h → Hadron (background)

Converted internally to 1 and 0 for model training

| Feature    | Description                                        |
| ---------- | -------------------------------------------------- |
| `fLength`  | Major axis of the ellipse (mm)                     |
| `fWidth`   | Minor axis of the ellipse (mm)                     |
| `fSize`    | 10-log of total pixel intensity (#photons)         |
| `fConc`    | Ratio of two brightest pixels to total intensity   |
| `fConc1`   | Ratio of brightest pixel to total intensity        |
| `fAsym`    | Asymmetry of light distribution along main axis    |
| `fM3Long`  | 3rd root of 3rd moment along major axis            |
| `fM3Trans` | 3rd root of 3rd moment along minor axis            |
| `fAlpha`   | Orientation angle of the ellipse (degrees)         |
| `fDist`    | Distance from image centroid to camera center (mm) |

In simple Words, this is a Binary Classification Problem Statement

The entire Math Notes is over here: https://bitspilaniac-my.sharepoint.com/:o:/g/personal/f20220910_pilani_bits-pilani_ac_in/EuHukMQUo5tMj6jh80K4A94BYCihpiKhlEehZ8STaECQWA?e=TUXJ8e