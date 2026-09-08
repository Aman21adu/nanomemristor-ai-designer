# Zero-Shot Cross-Technology Configuration Prediction for Memristive In-Memory AI Accelerators

## Overview

This project investigates whether accelerator-design knowledge learned from several memristor technologies can be transferred to a previously unseen memristor technology.

Instead of exhaustively testing every accelerator configuration for a new memristor device, the project trains a machine-learning predictor using known device technologies and attempts to recommend a near-optimal hardware configuration for a completely held-out technology.

The project combines:

- nanoscale memristor device properties from literature,
- memristive crossbar simulation,
- neural-network inference on MNIST,
- hardware design-space exploration,
- Random Forest regression,
- leave-one-technology-family-out zero-shot validation,
- and an interactive Streamlit demonstration.

---

# Research Question

> Can AI-hardware design knowledge learned from several nanoscale memristor technologies generalize well enough to predict a near-optimal accelerator configuration for a previously unseen memristor technology?

---

# Main Idea

The workflow is:

```text
Published memristor literature
        ↓
Device descriptors X
        ↓
Memristor-aware crossbar simulation
        ↓
Hardware configurations Y
        ↓
Simulated performance S
        ↓
Dataset (X, Y, S)
        ↓
Train predictor on known technologies
        ↓
Hide one complete technology family
        ↓
Predict performance of its configurations
        ↓
Recommend a near-optimal configuration
        ↓
Reveal exhaustive simulation
        ↓
Measure regret and search reduction