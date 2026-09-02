# Neutrino Waveform Classification

PyTorch CNN and regression networks for classifying and analyzing physics detector waveform data.
 
 ## What this is

 This is my final project for PHYS 41 (Scientific Computing in Python) at UC San Diego.

 ## What I built

- A PyTorch 'Dataset' class to load and normalize raw waveform signals with each one having 3,800 data points
- A 1D CNN that classifies whether a waveform looks like signal or background - got this one as an ROC AUC score of 0.856
- A separate regression network that predicts the energy of each event - got this one as a cosine similarity score of 0.86 between the predicted and true energy distributions

## Tools

PyTorch, NumPy, Matplotlib, scikit-learn

## Note

- Real detector data is noisy, so getting the normalization correct was actually more important than I thought going into it. Batch nornalization helped a lot with the classifiation model but apparently taking it out improved the regression model instead. It was a good exercise in seeing how the same data needs different handling based on if it is used for classifying or predicting a value.
- The dataset paths point to UCSD's internal Data Hub, so this won't run the same outside that environment.

