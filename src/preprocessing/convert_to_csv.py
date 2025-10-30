#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Optional


def detect_txt_file(directory: str) -> Optional[str]:
    candidates = [f for f in os.listdir(directory) if f.lower().endswith('.txt')]
    if not candidates:
        return None
    # Prefer the known dataset filename if present
    preferred = 'household_power_consumption.txt'
    for f in candidates:
        if f.lower() == preferred:
            return os.path.join(directory, f)
    # Fallback to the first .txt file
    return os.path.join(directory, candidates[0])


def convert_with_pandas(in_path: str, out_path: str) -> None:
    import pandas as pd

    # UCI dataset uses ';' delimiter and '?' for missing values.
    # We avoid dtype forcing to let pandas infer types; parse Date+Time if present.
    try:
        df = pd.read_csv(
            in_path,
            sep=';',
            na_values=['?'],
            low_memory=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read input with pandas: {e}")

    # If Date and Time columns exist, create a combined DateTime column
    if {'Date', 'Time'}.issubset(df.columns):
        try:
            df.insert(0, 'DateTime', pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce'))
        except Exception:
            # If parsing fails, skip adding DateTime
            pass

    df.to_csv(out_path, index=False)


def convert_streaming(in_path: str, out_path: str) -> None:
    # Generic converter: reads semicolon-separated .txt and writes comma-separated CSV.
    with open(in_path, 'r', encoding='utf-8', newline='') as fin, open(out_path, 'w', encoding='utf-8', newline='') as fout:
        # Try to sniff delimiter from first few KB
        sample = fin.read(4096)
        fin.seek(0)
        dialect = None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=';\t,|')
        except Exception:
            class _D(csv.Dialect):
                delimiter = ';'
                quotechar = '"'
                escapechar = None
                doublequote = True
                lineterminator = '\n'
                quoting = csv.QUOTE_MINIMAL
            dialect = _D

        reader = csv.reader(fin, dialect)
        writer = csv.writer(fout)

        # Replace '?' with empty string as missing value, pass through other fields
        for row in reader:
            writer.writerow(['' if v == '?' else v for v in row])


def main():
    parser = argparse.ArgumentParser(description='Convert the .txt dataset in this folder to CSV.')
    parser.add_argument('-i', '--input', help='Input .txt path (defaults to the first .txt found).')
    parser.add_argument('-o', '--output', help='Output .csv path (defaults to input name with .csv).')
    args = parser.parse_args()

    cwd = os.getcwd()
    in_path = args.input or detect_txt_file(cwd)
    if not in_path or not os.path.isfile(in_path):
        print('No .txt file found. Specify with --input.', file=sys.stderr)
        sys.exit(1)

    out_path = args.output or os.path.splitext(in_path)[0] + '.csv'

    # Try pandas for speed and type handling; fallback to streaming csv
    used_pandas = False
    try:
        import pandas as pd  # noqa: F401
        used_pandas = True
        convert_with_pandas(in_path, out_path)
    except Exception:
        used_pandas = False
        convert_streaming(in_path, out_path)

    print(f'Wrote CSV to: {out_path} (method: {"pandas" if used_pandas else "streaming"})')


if __name__ == '__main__':
    main()
