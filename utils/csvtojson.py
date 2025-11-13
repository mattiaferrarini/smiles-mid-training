# ...existing code...
import csv
import json
import os
import traceback
from pathlib import Path

def converti_csv_a_json(nome_file_csv, nome_file_json):
    dati_json = []
    try:
        csv_path = Path(nome_file_csv)
        json_path = Path(nome_file_json)

        # Diagnostics: print exact strings and existence checks immediately before opening
        print("Opening CSV path repr:", repr(str(csv_path)))
        print("Path.exists():", csv_path.exists())
        print("os.path.exists:", os.path.exists(str(csv_path)))
        print("Working directory:", os.getcwd())

        # Use Path.open for robustness and newline='' for csv
        with csv_path.open(mode='r', encoding='utf-8', newline='') as file_csv:
            lettore_csv = csv.DictReader(file_csv)
            for riga in lettore_csv:
                dati_json.append(riga)

        # Ensure parent exists for output
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open(mode='w', encoding='utf-8') as file_json:
            json.dump(dati_json, file_json, indent=4)
        print(f"Conversione completata: i dati sono stati salvati in '{json_path}'")
    except FileNotFoundError as fnf:
        print(f"Errore: Il file '{nome_file_csv}' non è stato trovato.")
        print("Exception repr:", repr(fnf))
        traceback.print_exc()
    except Exception as e:
        print(f"Si è verificato un errore durante la conversione: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent  # two levels up to project root
    csv_path = base / "csv" / "Periodic Table of Elements.csv"
    json_path = base / "json" / "Periodic Table of Elements.json"

    if not csv_path.exists():
        print("CSV non trovato (checked):", csv_path)
        print("Working directory:", os.getcwd())
    else:
        print("CSV trovato (checked):", csv_path)
        converti_csv_a_json(str(csv_path), str(json_path))
# ...existing code...