import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.undersampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
"""
on utilise Standard scaler pour transformer les donnees pour qu'ils 
soient dans la meme echelle sans grand ecart
et on doit aussi utiliser la technique de undersampling pour équilibrer les données
mais l'oversampling il ajoute des points de data par exemplle dans notre cas il prend 
2 vraies transactions frauduleueses similaires et calcule un point intermédiaire(moyenne pondérée) entre eux 
et crée un nouveaux point comme fraude plausible
"""
def load_and_preprocess(df,strategie ='undersampling'):
    print(f"Préprocessing des données-Stratégie : {strategie.upper()}")
    X = df.drop('Class',axis = 1)
    y = df['Class']
    
    print("Data set original: ")
    print(f"    -Total transactions: {len(df)}")
    print(f"    -Fraudes: {len(y[y==1])}")
    print(f"    -Non Fraudes: {len(y[y==0])}")
    """
    2.Split Temporel
    les données sont ordonnées par temps (Colonne Time) 70% training 30% test
    """
    train_size = int(len(df)*0.7)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    """
    3.Scaling(Standardisation)
    Scaling seulement pour les amounts car (V1-V28 ) sont déja scalées
    fit_transform pour les données d'entrainemet et transform pour les données de test (éviter la tricherie de modéle)
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
   
    X_train_scaled['Amount'] = scaler.fit_transform(X_train['Amount'])
    X_test_scaled['Amount'] = scaler.transform(X_test['Amount'])
    
    """
    4.Gestion du Déséquilibre
    choisir la strategie de gestion (undersampling vs oversampling)
    RandomUnderSampler supprime aléatoirement des exemples de la classe majoritaire
    pour l'oversampling il ajoute des exemples de la classe minoritaire (des données sythétiques)
    """
    if strategie == 'undersampling':
        #undersampling
        print("4.UNDERSAMPLING")
        print(" -Principe: Jeter des transactions non frauduleuses pour équilibrer")
        print(" -Résultats: 50%  fraudes , 50% normales ")
        
        under = RandomUnderSampler(random_state = 42)
        X_Train_final,y_train_final = under.fit_resample(X_train_scaled,y_train)
        print(f" -Avant {len(X_train_scaled)} transactions)")
        print(f" -Aprés {len(X_Train_final)} transactions)")
        print(f" -Economie de {len(X_train_scaled)-len(X_Train_final)} transactions)")
        
    
    elif strategie == 'oversampling':
        print("Oversampling avec SMOTE")
        print(" -Principe: créer des fraudes sythétiques entre vraies fraudes")
        print(" -Résultats: 50%  fraudes , 50% normales ")
        
        smote  = SMOTE(random_state = 42)
        X_train_final,y_train_final = smote.fit_resample(X_train_scaled,y_train)
        
        print(f" -Avant {len(X_train_scaled)} transactions)")
        print(f" -Aprés {len(X_train_final)} transactions)")
        print(f" -Nouvelles transactions frauduleuses crées: {len(y_train_final[y_train_final==1])-len(y_train[y_train==1])}") 
         
    