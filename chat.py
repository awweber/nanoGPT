import torch
import torch.nn as nn
from torch.nn import functional as F

MLPATH = './models/nano_gpt_shakespeare.pt'
DATAPATH = 'data/tinyshakespeare.txt'

# device configuration
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # M4 Check!


# --- 1. Daten Setup (Muss exakt wie im Training sein) ---
with open(DATAPATH, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device).unsqueeze(0) # Encoder für Chat-Input
decode = lambda l: ''.join([itos[i.item()] for i in l])

# --- 2. Transformer Modell laden und Inferenz starten ---
from llmlib import nanoGPT

# Modell instanziieren
model = nanoGPT.GPTLanguageModel()
model.to(device)

# Gewichte laden
try:
    model.load_state_dict(torch.load(MLPATH, map_location=device))
    model.eval() # Modell in den Evaluationsmodus setzen (wichtig wegen Dropout!)
    print("Modell erfolgreich geladen. Starte interaktiven Chat.")
except FileNotFoundError:
    print(f"FEHLER: '{MLPATH}' nicht gefunden.")
    exit()

# Interaktiver Chat Loop
print("-" * 50)
print("Starte den Shakespeare-Chatbot. Gib 'QUIT' ein, um zu beenden.")
print("-" * 50)

while True:
    try:
        prompt = input("Du: ")
        if prompt.upper() == 'QUIT':
            break

        # 1. Prompt encodieren
        context = encode(prompt)
        
        # 2. Generierung starten
        generated_tokens = model.generate(context, max_new_tokens=100)
        
        # 3. Das Ergebnis decodieren (wir ignorieren den Input)
        response = decode(generated_tokens[0])
        
        # Wir zeigen nur den generierten Teil (ab dem Ende des Prompts)
        response = response[len(prompt):] 
        print(f"Nano-GPT: {response}")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten (wahrscheinlich ein Zeichen, das nicht im Shakespeare-Datensatz war): {e}")