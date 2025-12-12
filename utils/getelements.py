import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


def _default_csv_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv', 'Periodic Table of Elements.csv'))


def _read_elements(csv_path):
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sym = (row.get('Symbol') or row.get('symbol') or '').strip()
            name = (row.get('Element') or row.get('Name') or row.get('element') or '').strip()
            num_raw = (row.get('AtomicNumber') or row.get('atomic_number') or row.get('Atomic Number') or '').strip()
            if not sym or not name or not num_raw:
                continue
            try:
                atomic_number = int(num_raw)
            except ValueError:
                continue

            rows.append({
                'symbol': sym,
                'name': name,
                'atomic_number': atomic_number,
            })
    return rows


def load_symbol_to_number(csv_path=None):
    """Return a vocabulary mapping element symbol -> atomic number (int).

    Args:
        csv_path: optional path to the periodic table CSV. When omitted, a default
            path `../csv/Periodic Table of Elements.csv` relative to this file
            is used.

    Returns:
        A dict mapping element symbol (e.g. 'H') to atomic number (int).
    """
    if csv_path is None:
        csv_path = _default_csv_path()

    rows = _read_elements(csv_path)
    return {r['symbol']: r['atomic_number'] for r in rows}


def load_name_to_symbol(csv_path=None):
    """Return a vocabulary mapping element name -> element symbol.

    Args:
        csv_path: optional path to the periodic table CSV.

    Returns:
        A dict mapping element name (e.g. 'Hydrogen') to its symbol (e.g. 'H').
    """
    if csv_path is None:
        csv_path = _default_csv_path()

    rows = _read_elements(csv_path)
    return {r['name']: r['symbol'] for r in rows}


def main():
    """CLI entrypoint: build symbol->atomic-number vocabulary and save as JSON.

    The output JSON will be written to `json/vocab_symbol_to_number.json` under the
    project root (two levels above this file) unless `--out` is provided.
    The vocabulary will include `"[UNK]": 0` and then element symbols mapped to
    their atomic numbers from the CSV.
    """
    parser = argparse.ArgumentParser(description='Save element symbol -> atomic number vocabulary as JSON')
    parser.add_argument('--csv', '-c', help='Path to Periodic Table CSV', default=None)
    parser.add_argument('--out', '-o', help='Output JSON path', default=None)
    parser.add_argument('--sort', choices=['atomic', 'symbol'], default='atomic', help='Sort output by atomic number or symbol')
    args = parser.parse_args()

    csv_path = args.csv or _default_csv_path()
    rows = _read_elements(csv_path)

    if args.sort == 'atomic':
        rows.sort(key=lambda r: int(r['atomic_number']))
    else:
        rows.sort(key=lambda r: r['symbol'])

    # Build vocabulary: include [UNK]: 0
    vocab: Dict[str, int] = {"[UNK]": 0}
    for r in rows:
        sym = r['symbol']
        num = int(r['atomic_number'])
        # ensure we don't overwrite [UNK]
        if sym in vocab:
            # skip or overwrite? we skip to preserve [UNK]
            continue
        vocab[sym] = num
    vocab_add ={ "[": 119, 
            "]": 120,
            "(": 121, 
            ")": 122,
            ".": 123, 
            "=": 124, 
            "#": 125,
            "-": 126,  
            "+": 127,
            "\\": 128,
            "/": 129,
            "@": 130, 
            "1": 131,
            "2": 132,
            "3": 133,
            "4": 134,
            "5": 135,
            "6": 136,
            "7": 137,
            "8": 138,
            "9": 139}
    
    vocab.update(vocab_add)

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        base = Path(__file__).resolve().parent.parent
        out_dir = base / 'json'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'vocab_symbol_to_number.json'

    # Save JSON
    with out_path.open('w', encoding='utf-8') as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)

    print(f"Saved vocabulary ({len(vocab)} tokens) to: {out_path}")


if __name__ == '__main__':
    main()


__all__ = ["load_symbol_to_number", "load_name_to_symbol"]