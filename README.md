# Credit Card Fraud Detection

## Description
Projet de machine learning pour détecter les fraudes bancaires comparant 3 modèles :
- **Logistic Regression** (modèle linéaire simple)
- **Random Forest** (ensemble d'arbres de décision)
- **Neural Network** (réseau de neurones avec Keras)

Dataset : 284,807 transactions dont 492 fraudes (0.17%) → déséquilibre traité par **oversampling (SMOTE)**

## Résultats attendus
| Modèle | F1-Score | ROC-AUC |
|--------|----------|---------|
| Logistic Regression | ~0.85 | ~0.97 |
| Random Forest | ~0.88 | ~0.97 |
| Neural Network | ~0.84 | ~0.97 |

## Execution

1. Installation locale
```bash
git clone https://github.com/votre-nom/credit-card-fraud-detection
cd credit-card-fraud-detection
pip install -r requirements.txt
python src/logistic_randomF.py
python src/neural_keras.py
````
2.Installation Docker (y'a des problémes)
````
docker build -t fraud-detection .
docker run --rm -v ${PWD}/results:/app/results fraud-detection
``
