<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p><strong>Projekti nga lënda:</strong> "Përgatitja dhe vizualizimi i të dhënave"</p>
      <p><strong>Profesor:</strong> PhD Mërgim Hoti</p>
      <p><strong>Studentët (Gr. 4):</strong></p>
      <ul>
        <li>Altin Morina</li>
        <li>Endri Binaku</li>
      </ul>
    </td>
  </tr>
</table>

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

## Project Execution

### Complete Execution Guide

For detailed step-by-step instructions on how to execute this entire project from start to finish, please refer to:

**[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)**

The execution guide provides comprehensive instructions for all 8 processing steps with detailed explanations of inputs,
outputs, and results.

### Quick Start

To execute the complete data processing pipeline:

1. **Prerequisites:**
   ```bash
   pip install pandas numpy
   ```

2. **Navigate to preprocessing directory:**
   ```bash
   cd src/preprocessing
   ```

3. **Execute all 8 steps in sequence:**
   ```bash
   python create_stratified_sample.py      # Step 1: Sampling
   python data_exploration.py              # Step 2: Exploration
   python data_quality_analysis.py         # Step 3: Quality Analysis
   python data_cleaning.py                 # Step 4: Data Cleaning
   python feature_engineering.py           # Step 5: Feature Engineering
   python data_aggregation.py              # Step 6: Data Aggregation
   python data_transformation.py           # Step 7: Data Transformation
   python feature_selection.py             # Step 8: Feature Selection
   ```

### Pipeline Overview

The execution pipeline transforms the data through these stages:

```
Original Dataset (2M rows, 127 MB)
    ↓ [Sampling]
Sample Dataset (1M rows, 50 MB)
    ↓ [Cleaning]
Cleaned Dataset (891K rows, 10 columns)
    ↓ [Feature Engineering]
Featured Dataset (891K rows, 37 columns)
    ↓ [Transformation & Selection]
Final Dataset (891K rows, ~35-40 columns)
```

### Expected Results

After complete execution, you will have:

- **5 main datasets** in `data/processed/`
- **7 aggregated views** in `data/aggregated/`
- **8 detailed reports** in `reports/`
- **Correlation matrix** in `outputs/`

**Total processing time:** ~5-10 minutes

**Important:** Always refer to the [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for detailed instructions, troubleshooting,
and verification steps.

## File Structure

```
│
├── src/preprocessing/          ← All scripts
│   ├── create_stratified_sample.py
│   ├── data_exploration.py
│   ├── data_quality_analysis.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_aggregation.py
│   ├── data_transformation.py
│   └── feature_selection.py
│
├── data/
│   ├── raw/                    ← Original dataset & sample
│   ├── processed/              ← Cleaned, featured, transformed, final
│   └── aggregated/             ← 7 aggregations
│
├── reports/
│   ├── analysis/               ← Exploration, features, aggregation, etc.
│   └── quality/                ← Quality, cleaning reports
│
├── outputs/                    ← Correlation matrix, figures
└── docs/                       ← README, guides
```

## Notes

- Large data files (`.txt`, `.csv`) are excluded from version control
- Use the sample dataset for development to avoid memory issues
- The original dataset is semicolon-delimited with `?` for missing values
- DateTime parsing available when using pandas in `convert_to_csv.py`

## Citation

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
