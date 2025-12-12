import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def _default_csv_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv', 'Periodic Table of Elements.csv'))


def _read_elements(csv_path):
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sym = (row.get('Symbol') or row.get('symbol') or '').strip()
            if not sym:
                continue

            rows.append({
                'symbol': sym,
            })
    return rows


def main():
    """CLI entrypoint: build character->index vocabulary and save as JSON.

    The output JSON will be written to `json/vocab_characters.json` under the
    project root (two levels above this file) unless `--out` is provided.
    The vocabulary will include `"[UNK]": 0` and then element symbols mapped to
    their atomic numbers from the CSV.
    """
    parser = argparse.ArgumentParser(description='Save element characters -> index vocabulary as JSON')
    parser.add_argument('--csv', '-c', help='Path to Periodic Table CSV', default=None)
    parser.add_argument('--out', '-o', help='Output JSON path', default=None)
    parser.add_argument('--sort', choices=['atomic', 'symbol'], default='atomic', help='Sort output by atomic number or symbol')
    args = parser.parse_args()

    csv_path = args.csv or _default_csv_path()
    rows = _read_elements(csv_path)

    # Build vocabulary: include [UNK]: 0
    vocab: Dict[str, int] = {"[UNK]": 0}
    index = 1
    for r in rows:
        sym = r['symbol']
        chars = list(sym)
        # ensure we don't overwrite [UNK]
        for char in chars:
            if char not in vocab:
                vocab[char] = index
                index += 1
    characters = list('[]().=#-+\\/@1234567890')
    vocab_add = {char: index + i for i, char in enumerate(characters) if char not in vocab}
    vocab.update(vocab_add)
    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        base = Path(__file__).resolve().parent.parent
        out_dir = base / 'json'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'vocab_characters.json'

    # Save JSON
    with out_path.open('w', encoding='utf-8') as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)

    print(f"Saved vocabulary ({len(vocab)} tokens) to: {out_path}")


if __name__ == '__main__':
    main()


