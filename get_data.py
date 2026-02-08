"""
Small script to get data for the project, it will download a dataset from kaggle
"""

import os
import subprocess
import zipfile
import requests

os.makedirs('data',exist_ok=True)

#get kaggle dataset from github no need for kaggle AUTH

url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"

print("Downloading dataset from github...")
response  =  requests.get(url)

if response.status_code == 200:
    with open('data/creditcard.csv', 'wb') as f:
        f.write(response.content)
    print("Dataset downloaded successfully!")
else:
    print("Failed to download dataset. Status code:", response.status_code)