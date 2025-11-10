import argparse
import csv
import os
import sys
from typing import Optional


def detect_txt_file(directory: str) -> Optional[str]:
    candidates = [f for f in os.listdir(directory) if f.lower().endswith('.txt')]
    if not candidates:
        return None
    preferred = 'household_power_consumption.txt'
    for f in candidates:
        if f.lower() == preferred:
            return os.path.join(directory, f)
    return os.path.join(directory, candidates[0])


def convert_with_pandas(in_path: str, out_path: str) -> None:
    import pandas as pd

    try:
        df = pd.read_csv(
            in_path,
            sep=';',
            na_values=['?'],
            low_memory=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read input with pandas: {e}")

    if {'Date', 'Time'}.issubset(df.columns):
        try:
            df.insert(0, 'DateTime', pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce'))
        except Exception:
            pass

    df.to_csv(out_path, index=False)


def convert_streaming(in_path: str, out_path: str) -> None:
    with open(in_path, 'r', encoding='utf-8', newline='') as fin, open(out_path, 'w', encoding='utf-8', newline='') as fout:
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

    used_pandas = False
    try:
        import pandas as pd
        used_pandas = True
        convert_with_pandas(in_path, out_path)
    except Exception:
        used_pandas = False
        convert_streaming(in_path, out_path)

    print(f'Wrote CSV to: {out_path} (method: {"pandas" if used_pandas else "streaming"})')


if __name__ == '__main__':
    main()
