FROM python:3.10-slim

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source
COPY get_data.py .
COPY requirements.txt .
COPY src/ ./src/
COPY data/ ./data/

# Créer les dossiers nécessaires
RUN mkdir -p models results

# Commande par défaut: exécuter les 3 scripts en séquence
CMD python get_data.py && \
    python src/logistic_randomF.py && \
    python src/neural_keras.py