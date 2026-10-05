# Detecting Optical Nonclassicality from Quantum Sampler Data

Code accompanying the Bachelor's thesis  
**“Detecting Optical Nonclassicality from Quantum Sampler Data”**  
by Rico Krahn, Friedrich Schiller University Jena.

## Data availability

This repository does not contain or redistribute experimental data from the
Paderborn Quantum Sampler (PaQS). The data were produced and published by the
authors of:

- M. Stefszky et al., *Benchmarking Gaussian and non-Gaussian input states
  with a hybrid sampling platform*  
  [Paper on arXiv](https://arxiv.org/abs/2512.08433)

- Official PaQS dataset:  
  [Zenodo dataset](https://doi.org/10.5281/zenodo.21639151)

The experimental data must be downloaded separately from the official Zenodo
repository.

## Related work

The model used in this thesis is based on the algebraic classifier introduced
by Martina Jung et al.:

- M. Jung et al., *Learning to detect optical nonclassicality*  
  [Paper on arXiv](https://arxiv.org/abs/2603.06319)

- Original implementation:  
  [MartinaJung/IdentifyingOpticalNonclassicality](https://github.com/MartinaJung/IdentifyingOpticalNonclassicality)

This thesis applies Jung et al.'s model to experimental photon-count data.
The main changes are:

- Martina's encoder already uses photon-number moments. Here they are computed
  directly from histogram counts instead of shot-level inputs. Training uses a
  zero-margin loss, and the polynomial skeleton builds only the terms needed.
- Separate interpolation and extrapolation builders turn the raw TXT files into
  train and test HDF5 datasets. They combine ten accepted 0.1 s files per bin,
  then skip five accepted files before the next bin. The 0.5 s gap is nominal
  when raw files are missing. Oversampling is applied only to training data.
- Complete acquisition periods from the training file are held out for
  validation. The best validation checkpoint is selected before the test file
  is evaluated.
