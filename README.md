# Spiking Poisson Diffusion for Audio Denoising

A diffusion-inspired audio denoising framework implemented using Spiking Neural Networks (SNNs) and Poisson spike-based stochastic corruption.

---

## Overview

This project explores whether temporal spike-based neural dynamics can learn a diffusion-style denoising objective for audio waveforms.

The implementation combines:

- Poisson spike noise
- timestep-conditioned denoising
- temporal spike processing
- diffusion-inspired noise prediction

The model is trained to estimate corruption added to an audio waveform at arbitrary diffusion timesteps and reconstruct a cleaner version of the signal.

---

## Motivation

Diffusion models learn to reverse stochastic corruption processes through iterative denoising objectives.

Spiking Neural Networks provide a biologically inspired computational framework based on discrete spike events, membrane dynamics, and temporal processing.

This project investigates the interaction between these two paradigms by applying spike-based computation to a diffusion-style denoising task.

---

## Method Summary

### Forward Corruption Process

The forward process applies:

1. timestep-dependent signal decay
2. Poisson spike stochasticity
3. Gaussian perturbation

to generate progressively corrupted audio waveforms.

The model is trained to predict:

\[
\epsilon = x_t - x_0
\]

where:

- \(x_0\) = clean waveform
- \(x_t\) = corrupted waveform

---

### Spiking Neural Network

The denoiser includes:

- timestep embeddings
- rate-based spike encoding
- Leaky Integrate-and-Fire (LIF) neurons
- temporal membrane integration

The network receives noisy waveforms and predicts the injected corruption.

---

## Training Objective

The training pipeline follows:

clean waveform  
→ forward corruption  
→ noisy waveform  
→ spike encoding  
→ temporal SNN processing  
→ noise prediction loss

Mean Squared Error (MSE) is used as the optimization objective.

---

## Dataset

The implementation uses the SpeechCommands dataset from torchaudio.

Preprocessing includes:

- resampling
- mono conversion
- waveform normalization
- fixed-length waveform generation

---

## Project Structure

```text
.
├── notebook.ipynb
├── data/
└── README.md
```

---

## Environment

### Main Dependencies

- PyTorch
- torchaudio
- snnTorch
- NumPy
- Matplotlib

Install dependencies using:

```bash
pip install torch torchaudio snntorch matplotlib numpy soundfile
```

---

## Running the Notebook

Open the notebook and run cells sequentially:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

---

## Current Scope

This implementation focuses on:

- noise prediction
- denoising behavior
- temporal spike processing

The notebook does not implement a full iterative diffusion sampling pipeline.

---

## Possible Extensions

- convolutional SNN architectures
- spectrogram-based denoising
- iterative reverse diffusion
- alternative spike encoding schemes
- larger training datasets
- neuromorphic deployment

---

## Notes

This repository is intended as an experimental exploration of diffusion-style denoising using spike-based neural computation. The implementation emphasizes interpretability and temporal spike dynamics over large-scale generative performance.
