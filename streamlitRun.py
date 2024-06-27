import subprocess
import requests
import gdown
import streamlit

# Download the model from Google Drive
file_id = '1L69so1We0E67X0bRHSOkEsvPYeSIEnr6'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
output = 'trained_model.h5'
gdown.download(url, output, quiet=False)

# Start the main app
subprocess.run(["python", "app.py"])
