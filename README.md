# Neutrino Waveform Classification

PyTorch CNN and regression networks for classifying and analyzing physics detector waveform data.

## What this is

This is my final project for PHYS 41 (Scientific Computing in Python) at UC San Diego.

## What I built

- A PyTorch `Dataset` class to load and normalize raw waveform signals, each one 3,800 data points long
- A 1D CNN that classifies whether a waveform looks like signal or background — ROC AUC score of 0.856
- A separate regression network that predicts the energy of each event — cosine similarity score of 0.87 between the predicted and true energy distributions

## Tools

PyTorch, NumPy, Matplotlib, scikit-learn

## Notes and known limitations

- Real detector data is noisy, so getting the normalization correct was more important than I expected going in. Batch normalization helped a lot with the classification model, but removing it actually improved the regression model instead — a good example of how the same data needs different handling depending on whether it's used for classifying or predicting a value.
- Normalization stats (mean/std) are currently computed per-dataset rather than fit on training data and applied to test — a cleaner version would fix this so the test set isn't normalized using its own statistics.
- Training loss is noisy, especially for the regression model in later epochs (it spikes well above its typical range around epoch 27–28 of 30 before settling). There's no validation-based checkpointing, so the reported metric reflects whatever the final epoch produced. A cleaner version would add learning-rate scheduling and checkpoint on a validation metric instead.
- The dataset paths point to UCSD's internal Data Hub and won't resolve outside that environment. That environment's course data has since been decommissioned, so this notebook can no longer be re-run or re-verified — the results and outputs reflect the original run made while the data was available.
