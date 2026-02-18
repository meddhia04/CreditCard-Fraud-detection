from sklearn.metrics import f1_score,roc_auc_score,classification_report

#la fonction qui va nous chercher le bon seuil de décision au lieu de 0.5
def best_thresh(model,X_test,y_test):
    #si c'est un model sklearn
    if hasattr(model,'predict_proba'):
        y_proba = model.predict_proba(X_test)[:,-1]
    else:
        #c'est un model keras
        y_proba = model.predict(X_test,verbose=0).flatten()
        
    
    #chercher le meilleur seuil pour maximiser le F1-Score
    best_thresh = 0.5
    best_f1 = 0
    
    for thresh in [i/100 for i in range(10,91)]:#seuils de 0.1 a 0.9
        y_pred_thresh = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test,y_pred_thresh)
        if f1>best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh