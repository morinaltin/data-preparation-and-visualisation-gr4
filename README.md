# Individual Household Electric Power Consumption

Analysis and preprocessing of household electric power consumption dataset.

## Dataset

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Period**: December 2006 - November 2010 (47 months)
- **Measurements**: ~2 million minute-by-minute recordings
- **Size**: ~127 MB (txt), ~138 MB (csv)

### Variables

| Variable | Description | Unit |
|----------|-------------|------|
| Date | Date in format dd/mm/yyyy | - |
| Time | Time in format hh:mm:ss | - |
| Global_active_power | Household global minute-averaged active power | kilowatt |
| Global_reactive_power | Household global minute-averaged reactive power | kilowatt |
| Voltage | Minute-averaged voltage | volt |
| Global_intensity | Household global minute-averaged current intensity | ampere |
| Sub_metering_1 | Energy sub-metering No. 1 (kitchen) | watt-hour |
| Sub_metering_2 | Energy sub-metering No. 2 (laundry) | watt-hour |
| Sub_metering_3 | Energy sub-metering No. 3 (climate control) | watt-hour |

**Note**: Missing values are coded as `?`.

## Setup

### Requirements

- Python 3.7+
- Optional: pandas (for faster CSV conversion)

### Getting the Dataset

1. Download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
2. Extract `household_power_consumption.txt` to this directory
3. The dataset files are excluded from git (see `.gitignore`)

## Usage

### Convert TXT to CSV

Convert the semicolon-delimited `.txt` file to standard CSV format:

```bash
py convert_to_csv.py
```

**Options:**
- `-i, --input`: Specify input file path
- `-o, --output`: Specify output file path

### Create Sample Dataset

Generate a smaller sample for development/testing:

```bash
py create_sample.py
```

This creates `household_power_consumption_sample.txt` with ~10,000 data points (every 100th line from the original ~2M rows).

**Options:**
- `-i, --input`: Input file (default: `household_power_consumption.txt`)
- `-o, --output`: Output file (default: `household_power_consumption_sample.txt`)
- `-n, --num-lines`: Number of lines to include (default: 10000)
- `-s, --skip`: Sampling rate - take every Nth line (default: 100)

**Examples:**

```bash
# Create smaller sample (5000 lines, every 200th row)
py create_sample.py -n 5000 -s 200

# Create larger sample (50000 lines, every 20th row)
py create_sample.py -n 50000 -s 20

# Sample from CSV instead
py create_sample.py -i household_power_consumption.csv -o sample.csv
```

## File Structure

```
.
├── README.md                                    # This file
├── .gitignore                                   # Excludes large data files
├── convert_to_csv.py                            # TXT → CSV converter
├── create_sample.py                             # Sample dataset generator
├── household_power_consumption.txt              # Original dataset (not in git)
├── household_power_consumption.csv              # Converted CSV (not in git)
└── household_power_consumption_sample.txt       # Sample dataset (git-friendly)
```

## Notes

- Large data files (`.txt`, `.csv`) are excluded from version control
- Use the sample dataset for development to avoid memory issues
- The original dataset is semicolon-delimited with `?` for missing values
- DateTime parsing available when using pandas in `convert_to_csv.py`

## Citation

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
