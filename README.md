# CreditCard-Fraud-detection

ML project to detect fraudulent transactions using Logistic Regression, Random Forest, and Neural Networks on imbalanced data.
Dataset

Kaggle Credit Card dataset - 284,807 transactions with only 492 frauds (0.17%)
-> Tech Stack

    Python, Pandas, NumPy

    Scikit-learn, TensorFlow/Keras

    Imbalanced-learn (SMOTE)

    Docker 🐳

-> Features

    Temporal train/test split (70/30)

    2 imbalance strategies: undersampling, SMOTE

    Threshold optimization for F1-score

    Model comparison with ROC curves & confusion matrices
->Results
Model	F1-Score	ROC-AUC
Random Forest	~0.85	>0.97
Neural Network	~0.82	>0.97
  
