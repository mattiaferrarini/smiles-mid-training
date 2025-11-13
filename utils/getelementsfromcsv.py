import os
import csv
import argparse
from typing import Dict, List, Any, Optional


def _default_csv_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv', 'Periodic Table of Elements.csv'))


def _read_elements(csv_path: str) -> List[Dict[str, Any]]:
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


def load_symbol_to_number(csv_path: Optional[str] = None) -> Dict[str, int]:
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


def load_name_to_symbol(csv_path: Optional[str] = None) -> Dict[str, str]:
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


def main() -> None:
    """CLI entrypoint: print symbol -> atomic number vocabulary.

    Usage examples:
        python .\\utils\\getelements.py
        python .\\utils\\getelements.py --csv .\\csv\\Periodic Table of Elements.csv --sort symbol
    """
    parser = argparse.ArgumentParser(description='Print element symbol -> atomic number vocabulary')
    parser.add_argument('--csv', '-c', help='Path to Periodic Table CSV', default=None)
    parser.add_argument('--sort', choices=['atomic', 'symbol'], default='atomic', help='Sort output by atomic number or symbol')
    args = parser.parse_args()

    csv_path = args.csv or _default_csv_path()
    rows = _read_elements(csv_path)

    if args.sort == 'atomic':
        rows.sort(key=lambda r: int(r['atomic_number']))
    else:
        rows.sort(key=lambda r: r['symbol'])

    for r in rows:
        print(f"{r['symbol']}: {r['atomic_number']}")


if __name__ == '__main__':
    main()


__all__ = ["load_symbol_to_number", "load_name_to_symbol"]