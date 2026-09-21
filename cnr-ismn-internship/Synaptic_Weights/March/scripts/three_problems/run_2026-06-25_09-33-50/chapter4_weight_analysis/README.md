# Chapter 4 weight-matrix analysis

## What is included
- Histogram-based Shannon marginal entropy of effective weight values.
- Entropy sensitivity for 32, 64, and 128 common bins per layer.
- Row, column, flattened, and 2D matrix autocorrelation.
- Singular-value spectra, shuffled matrix controls, stable rank, effective rank,
  and top-k spectral-energy fractions.
- Pairwise correlation and normalized Frobenius distance between the three
  task-specific effective matrices.

## What is not included
Fisher information cannot be calculated from saved weights alone. An empirical
Fisher calculation requires a fixed dataset of state-transition samples and
per-sample gradients of a specified TD loss with respect to the parameters.

## Interpretation limits
- These checkpoints are final snapshots. They do not show how statistics evolve
  through training.
- Shuffled controls preserve the marginal weight distribution but destroy index
  organization. They are an exploratory random-control comparison, not a full
  Marchenko--Pastur fit.
- Matrix-index autocorrelation is a statistical descriptor. Hidden-neuron
  permutation invariance means it is not direct physical spatial correlation.

## Layers
- FC.0.weight
- FC.2.weight
- FC.4.weight
- FC.6.weight
