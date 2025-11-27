import yaml
from pathlib import Path
import os
from huggingface_hub import login
from dotenv import load_dotenv # 

def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)

def hf_auth():
    load_dotenv() 
    token = os.environ.get("HF_TOKEN")
    
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print("Hugging Face authentication successful (using HF_TOKEN).")
        except Exception as e:
            print(f"Warning: Hugging Face authentication failed (Token non valido): {e}")
    else:
        try:
            login(add_to_git_credential=False)
            print("Warning: HF_TOKEN non trovato nell'ambiente, uso l'autenticazione tramite cache locale.")
        except Exception:
            print("Warning: Nessuna autenticazione Hugging Face trovata. Accesso come utente anonimo.")
            
