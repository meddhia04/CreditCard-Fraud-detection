import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
exploration du data qu'on va travailler avec
"""
df = pd.read_csv('../data/creditcard.csv')
print("Data Shape:(lignes,colonnes) dans le fichier CSV ",end="")
print(df.shape)
print("Some snippets of the data: ")
print(df.head())

#Analyse de la distribution des classes
print("Distributions des classes:",end="")
fraud_count = df['Class'].value_counts()
print(f"Non fraude: {fraud_count[0]}, ({fraud_count[0]/df.shape[0]*100:.2f}%) ")#le calcule est pour qu'on calcule en pourcentage avec deux chiffres decimales
print(f"Fraude: {fraud_count[1]}, ({fraud_count[1]/df.shape[0]*100:.2f}%) ")
#Visualisation de déséquilibre
plt.figure(figsize=(10,6))
sns.countplot(x='Class',data = df)#seaborn draws the plot , uses matplotlib internally
plt.title("Distribution des transactions")
plt.show()
#plt.savefig("../results/distribution.png")