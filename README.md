# Gender Detection Project
Project for the course Machine Learning And Pattern Recognition - Politecnico di Torino - year 2022
## Launch the project
```{python}
python3 main.py
```
## Files Content at `src/`
- `Dataset/INFO.txt`: general information about the dataset
- `Dataset/Train.txt`: trainig set (one sample per raw) - 3000 samples of class Male and 3000 samples of class Female
- `Dataset/Test.txt`: evaluation set (one sample per raw) - 2000 samples of class Male and 2000 samples of class Female
- `Constants.py`: constants like number of samples, numer of features
- `DataImport.py`: functions to read the data from `Dataset/` folder to store them in `numpy` objects
- `DimensionalityReduction.py`: implements PCA and LDA dimensionality reduction techniques
- `GaussianClassifiers.py`: implements the Generative Gaussian Models (MVG Full Covariance, MVG Tied Covariance, Naive Bayes)
- `GMMClassifier.py`: implements the GMM model for classification (LBG and EM algorithm are used)
- `LDAClassifier.py`: implements LDA method as linear classification model (equivalent to Tied Gaussian classifier)
- `LogisticRegressionClassifier.py`: implements Linear Logistic Regression and Quadratic Logistic Regression models
- `SVMClassifier.py`: implements Linear SVM and Kernel SVM (polynomial and RBF kernels)
- `main.py`: it contains all the calls to the functions used to train and evaluate the different methods
- `ModelEvaluation.py`: implements all the functions used to evaluate the models (k-fold cross validation, computation of DCF and minDCF,
  ROC curve, Bayes Error Plot, score calibration)
- `PreProcessing.py`: implements Z-Normalization and Gaussianization pre processing techniques
  
## Design
All the classification models need to be instatiated as objects of their class by passing to the constructor all the model hyperparameters
needed for the training phase and each one provides these methods:
- `train(DT, LT)`: to learn model parameters from the Training Set (DT: samples of training set, LT: class labels)
- `predict(DE)`: samples of the Testing Set are scored using Log-Likelihood ratio (DE: samples to be labelled)
