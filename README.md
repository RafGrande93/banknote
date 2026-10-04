# Banknote authentication using machine learning

This project uses classification algorithms to verify if some banknotes are genuine or false based on numerical features extracted from banknote images, such as variance, kurtosis, skewness and entropy

## Models
The following classification algorithms are compared:

- Logistic Regression (baseline model)
- K-Nearest Neighbors classification 
- Support Vector Classifier (linear and nonlinear kernel)
- Random Forest 

Hyperparameters are tuned using `GridSearchCV`, and model performance is evaluated using accuracy, precision, recall, and F1 score.

## Dataset
The dataset contains four numerical features obtained from wavelet-transformed banknote images:

- Variance
- Skewness
- Kurtosis
- Entropy

Data cleaning, data visualization and rescaling was performed to understand better the characteristics of these parameters.

## Technologies
- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Results
All tested models achieved high classification performance, with KNN and SVC obtaining the best results on the test set.
