# Neutrino Waveform Classification

This is an archived course project (PHYS 41 at UC San Diego). Its data is no longer available, so the results are from the original run and cannot be re-run.

PyTorch CNN and regression networks for classifying and analyzing physics detector waveform data.

## What this is

This is my final project for PHYS 41 (Scientific Computing in Python) at UC San Diego. The waveforms come from the Majorana Demonstrator, an experiment searching for neutrinoless double-beta decay.

## What I built

- A PyTorch `Dataset` class to load and normalize raw waveform signals, each one 3,800 data points long
- A 1D CNN that classifies whether a waveform looks like signal or background, with a ROC AUC of 0.857 on the test set
- A separate regression network that predicts the energy of each event. Its score of 0.87 is the cosine similarity between the histograms of the predicted and true energies (bins of width 5 from 100 to 4000). That compares the shapes of the two distributions, not the accuracy for each event

## Tools

PyTorch, NumPy, Matplotlib, scikit-learn

## Notes and known limitations

- Real detector data is noisy, so getting the normalization correct was more important than I expected going in. Batch normalization helped a lot with the classification model, but removing it actually improved the regression model instead. The same data needed different handling for classifying and for predicting a value.
- Normalization stats (mean/std) are currently computed per-dataset rather than fit on training data and applied to test. A cleaner version would fix this so the test set isn't normalized using its own statistics.
- Training loss is noisy, especially for the regression model in later epochs (it spikes well above its typical range around epochs 27 and 28 of 30 before settling). There's no validation-based checkpointing, so the reported metric reflects whatever the final epoch produced. A cleaner version would add learning-rate scheduling and checkpoint on a validation metric instead.
- The test loader has `drop_last=True` and `shuffle=True`, so the AUC is computed on 9,984 of the 10,000 test events (312 batches of 32). Which 16 events are left out changes from run to run.
- The dataset paths point to UCSD's internal Data Hub and won't resolve outside that environment. That environment's course data has since been decommissioned, so this notebook can no longer be re-run or re-verified. The results and outputs reflect the original run made while the data was available.
