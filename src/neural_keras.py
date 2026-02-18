# neural_keras.py - VERSION CORRIGÉE
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import os
import pandas as pd
from preprocessing import load_and_preprocess

def create_neural_network(input_dim):
    """Modèle neural network batch normalization c'est pour pour le remttre a l'echelle"""
    
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

def train_neural_network(X_train, y_train, X_test, y_test):
    """Entraînement du réseau de neurones"""
    print("1;TRAINING NEURAL NETWORK")
    
    # Création modèle
    model = create_neural_network(X_train.shape[1])
    
    # Calcul poids des classes
    fraud_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
    #punir si la fonction prédire que c'est une transaction normale alors que c'est une  fraude
    class_weight = {0: 1., 1: fraud_ratio}
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )
    
    # PRÉDICTIONS - C'EST ICI QU'ON OBTIENT y_proba
    y_proba = model.predict(X_test, verbose=1).flatten()
    
    # Trouver le meilleur seuil avec Precision-Recall(qui test tous les seuls possible)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)#formule de F1-Score
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n Meilleur seuil: {best_th:.4f} (F1 optimal = {best_f1:.4f})")
    
    # Test à différents seuils
    print(f"\n Analyse des seuils:")
    print(f"{'Seuil':<10} {'Prédictions+':<15} {'Fraudes trouvées':<20} {'F1-Score':<10}")
    print("-" * 55)
    
    for seuil in [0.5, 0.3, 0.1, 0.05, 0.01, best_th]:
        y_pred = (y_proba >= seuil).astype(int)
        fraudes_trouvees = ((y_pred == 1) & (y_test == 1)).sum()
        f1 = f1_score(y_test, y_pred)
        print(f"{seuil:<10.4f} {y_pred.sum():<15} {fraudes_trouvees}/{y_test.sum():<20} {f1:<10.4f}")
    
    # Prédictions finales avec meilleur seuil
    y_pred_final = (y_proba >= best_th).astype(int)
    f1_final = f1_score(y_test, y_pred_final)
    roc_auc = roc_auc_score(y_test, y_proba)
    print("Keras Modle Results: ")
    print(f"   - F1-Score: {f1_final:.4f}")
    print(f"   - ROC-AUC: {roc_auc:.4f}")
    
    return model, history, y_pred_final, y_proba, {
        'f1': f1_final,
        'roc_auc': roc_auc,
        'best_threshold': best_th
    }

if __name__ == "__main__":
    # Chargement données
    df = pd.read_csv("../data/creditcard.csv")
    
    # Preprocessing
    X_train, y_train, X_test, y_test = load_and_preprocess(df, 'oversampling')
    # Entraînement
    model, history, y_pred, y_proba, metrics = train_neural_network(
        X_train, y_train, X_test, y_test
    )