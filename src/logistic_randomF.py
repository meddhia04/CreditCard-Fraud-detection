import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,f1_score)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_preprocess
from thresh_hold_opt import best_thresh
import os
"""
LE ROC-AUC évalue la capacité de sépération,F1-SCORE évalue prédictions finales
Logistic regression travaille bien sur les relations linéaires et il est facile a interprété
Random Forest travaille avec les relations non-linéaire il a une bonne accuracy , mais il est difficile a interpreté
Le Random Forest fonctionnement : crée plusieurs arbres de décision a partir de données légérement différentes , chaque arbre fait sa propre prédiction,
aprés le modéle combine les résultats si classification -> vote majoritaire si régression-> Moyenne des prédictions
"""
def train_sklearn_models(X_train,y_train,X_test,y_test):
    #créer le dossier resuls
    os.makedirs("../results",exist_ok=True)
    results = {}
    #1.Logistic Regression
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter = 1000,
        solver='saga',#pour les dataset grand comme celle qu'ona creditcard.csv
        random_state = 42,
        n_jobs=-1
    )
    lr.fit(X_train,y_train)
    y_pred_lr = lr.predict(X_test)#renvoie directement la classe prédite
    #best thresh
    best_th = best_thresh(lr,X_test,y_test)
    y_proba_pred_lr = lr.predict_proba(X_test)[:,1]#renovoie les provabilité de chaque classe
    y_pred_lr_thresh = (y_proba_pred_lr >= best_th).astype(int)
    #stocker les résultats de logistic regression
    results['Logistic Regression'] = {
        'model':lr,
        'y_pred': y_pred_lr,
        'y_proba': y_proba_pred_lr,
        'f1': f1_score(y_test,y_pred_lr_thresh),
        'roc_auc': roc_auc_score(y_test,y_proba_pred_lr),#test capacité de séparation de class(test tout les seuils automatiquement)
        'report': classification_report(y_test,y_pred_lr_thresh)
    }
    print("Logistic Regression F1-SCORE and ROC-AUC",end="")
    print(f"->F1-SCORE:{results['Logistic Regression']['f1']:.4f} / ",end="")
    print(f"ROC-AUC:{results['Logistic Regression']['roc_auc']:.4f}")
    #sauvegarde de model
    #joblib.dump(lr,'../models/logistic_regression.pkl')
    #2.RANDOM FORREST
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)
    #chercher le bon thresh
    best_th = best_thresh(rf,X_test,y_test)
    print(f"best threshold = {best_th} ")
    y_proba_rf = rf.predict_proba(X_test)[:,1]
    y_pred_rf_thresh = (y_proba_rf >= best_th).astype(int)
    
    #stocker les résultats de random forest
    results['Random Forest'] = {
        'model':rf,
        'y_pred':y_pred_rf_thresh,
        'y_proba':y_proba_rf,
        'f1': f1_score(y_test,y_pred_rf_thresh),
        'roc_auc': roc_auc_score(y_test,y_proba_rf),
        'best_threshold' : best_th,
        'report':classification_report(y_test,y_pred_rf)
    }
    print("Random Forest Model Results: ",end="")
    print(f"->F1-SCORE: {results['Random Forest']['f1']:.4f} / ",end="")
    print(f"ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")
    return results
    
    #Feature importance , les feature les plus important
# 3. MATRICE DE CONFUSION
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.savefig(f'../results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    
if __name__ == '__main__':
    df = pd.read_csv("../data/creditcard.csv")
    #je veux utiliser la stratégie d'oversampling
    X_train,y_train,X_test,y_test = load_and_preprocess(df,'oversampling')
    r = train_sklearn_models(X_train,y_train,X_test,y_test)
    plot_confusion_matrix(y_test,r['Random Forest']['y_pred'],"Random Forest")
    
    
