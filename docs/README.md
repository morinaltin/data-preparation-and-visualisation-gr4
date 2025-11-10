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

### Original Variables

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

## Project Structure

This project is divided into two phases:

- **Phase 1 (Completed)**: Data preprocessing, cleaning, feature engineering, transformation, and feature selection
- **Phase 2 (In Progress)**: Advanced outlier detection, false positive/negative analysis, and multivariate exploration

## Setup

### Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn (for Phase 2)

Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

### Getting the Dataset

1. Download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
2. Extract `household_power_consumption.txt` to `data/raw/` directory
3. The dataset files are excluded from git (see `.gitignore`)

## Phase 1: Data Preprocessing - Column Evolution

This section documents how the dataset columns changed through each preprocessing step.

### Step 1: Sampling
**Script**: `src/preprocessing/create_stratified_sample.py`

**Input**: `data/raw/household_power_consumption.txt` (2,075,259 rows × 9 columns)
**Output**: `data/raw/household_power_consumption_sample.txt` (999,970 rows × 9 columns)

**Columns**: No changes
- Date, Time, Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3

**Result**: 48.2% of original data (stratified sampling)

---

### Step 2: Data Exploration
**Script**: `src/preprocessing/data_exploration.py`

**Input**: `data/raw/household_power_consumption_sample.txt` (999,970 rows × 9 columns)
**Output**: Reports only (read-only analysis)

**Columns**: No changes to dataset
**Output Files**:
- `reports/analysis/exploration_report.txt`
- `reports/analysis/exploration_statistics.csv`
- `reports/analysis/exploration_sample.csv`

**Analysis**: Data types, summary statistics, value ranges, time period analysis

---

### Step 3: Data Quality Analysis
**Script**: `src/preprocessing/data_quality_analysis.py`

**Input**: `data/raw/household_power_consumption_sample.txt` (999,970 rows × 9 columns)
**Output**: Reports only (read-only analysis)

**Columns**: No changes to dataset
**Output Files**:
- `reports/quality/quality_report.txt`
- `reports/quality/quality_missing_values.csv`
- `reports/quality/quality_outliers.csv`

**Findings**:
- Missing values: 87,731 (0.97%) - all 7 numeric columns affected simultaneously
- Duplicates: 0
- Outliers detected: 256,973 (25.7% using IQR method with 1.5×IQR)

---

### Step 4: Data Cleaning
**Script**: `src/preprocessing/data_cleaning.py`

**Input**: `data/raw/household_power_consumption_sample.txt` (999,970 rows × 9 columns)
**Output**: `data/processed/household_power_consumption_cleaned.csv` (891,357 rows × 10 columns)

**Column Changes**:
- **Added**: `DateTime` (datetime) - Integration of Date and Time columns
- **Removed**: None (Date and Time columns retained)
- **Modified**: All numeric columns (missing values filled via time-based interpolation)

**Operations**:
1. Created `DateTime` column from Date + Time
2. Filled 87,731 missing values using linear interpolation (time-based)
3. Removed 108,613 outlier rows (10.86%) using IQR method (3×IQR threshold)
4. Clipped negative values to 0

**Final Columns** (10):
- DateTime, Date, Time
- Global_active_power, Global_reactive_power, Voltage, Global_intensity
- Sub_metering_1, Sub_metering_2, Sub_metering_3

**Output Files**:
- `data/processed/household_power_consumption_cleaned.csv`
- `reports/quality/cleaning_report.txt`

---

### Step 5: Feature Engineering
**Script**: `src/preprocessing/feature_engineering.py`

**Input**: `data/processed/household_power_consumption_cleaned.csv` (891,357 rows × 10 columns)
**Output**: `data/processed/household_power_consumption_with_features.csv` (891,357 rows × 34 columns)

**Column Changes**:
- **Added**: 24 new features
- **Removed**: None

**New Features by Category**:

1. **Temporal Features** (9 columns):
   - `Year` (int) - Year from DateTime
   - `Month` (int) - Month (1-12)
   - `Day` (int) - Day of month
   - `Hour` (int) - Hour (0-23)
   - `Minute` (int) - Minute (0-59)
   - `DayOfWeek` (int) - Day of week (0=Monday, 6=Sunday)
   - `DayName` (str) - Day name (Monday-Sunday)
   - `MonthName` (str) - Month name (January-December)
   - `WeekOfYear` (int) - Week number (1-53)

2. **Binary Features** (5 columns):
   - `IsWeekend` (int) - 1 if Saturday/Sunday, 0 otherwise
   - `IsNight` (int) - 1 if hour 22-5, 0 otherwise
   - `IsMorning` (int) - 1 if hour 6-11, 0 otherwise
   - `IsAfternoon` (int) - 1 if hour 12-17, 0 otherwise
   - `IsEvening` (int) - 1 if hour 18-21, 0 otherwise

3. **Categorical Features** (2 columns):
   - `Season` (str) - Winter/Spring/Summer/Autumn
   - `TimeOfDay` (str) - Morning/Afternoon/Evening/Night

4. **Calculated Features** (4 columns):
   - `Sub_metering_4` (float) - Unmeasured energy (Global_active_power × 1000/60 - sum of Sub_metering_1,2,3)
   - `Total_Sub_metering` (float) - Sum of all sub-metering values
   - `Energy_per_minute` (float) - Global_active_power / 60 (kWh per minute)
   - `Intensity_ratio` (float) - Global_intensity / (Voltage / 1000)

5. **Statistical Features** (4 columns):
   - `Power_1h_avg` (float) - Rolling average of Global_active_power (60 minutes window)
   - `Power_24h_avg` (float) - Rolling average of Global_active_power (1440 minutes window)
   - `Power_prev_1h` (float) - Lag feature: Global_active_power from 1 hour ago
   - `Power_change_1h` (float) - Change in power from previous hour

**Final Columns** (34):
- Original: DateTime, Date, Time, Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
- New: Year, Month, Day, Hour, Minute, DayOfWeek, DayName, MonthName, WeekOfYear, IsWeekend, IsNight, IsMorning, IsAfternoon, IsEvening, Season, TimeOfDay, Sub_metering_4, Total_Sub_metering, Energy_per_minute, Intensity_ratio, Power_1h_avg, Power_24h_avg, Power_prev_1h, Power_change_1h

**Output Files**:
- `data/processed/household_power_consumption_with_features.csv`
- `reports/analysis/features_report.txt`

---

### Step 6: Data Aggregation
**Script**: `src/preprocessing/data_aggregation.py`

**Input**: `data/processed/household_power_consumption_with_features.csv` (891,357 rows × 34 columns)
**Output**: 7 aggregated datasets (separate files)

**Column Changes**: Creates separate aggregated views (original dataset unchanged)

**Aggregated Datasets**:

1. **aggregation_daily.csv**: Daily aggregates (mean, sum, min, max per day)
2. **aggregation_hourly.csv**: Hourly aggregates (mean, sum per hour)
3. **aggregation_weekly.csv**: Weekly aggregates (mean, sum per week)
4. **aggregation_monthly.csv**: Monthly aggregates (mean, sum per month)
5. **aggregation_seasonal.csv**: Seasonal aggregates (mean, sum per season)
6. **aggregation_timeofday.csv**: Time of day aggregates (mean, sum by time period)
7. **aggregation_hour_weekend.csv**: Hourly aggregates split by weekend/weekday

**Output Files**:
- `data/aggregated/aggregation_*.csv` (7 files)
- `reports/analysis/aggregation_report.txt`

---

### Step 7: Data Transformation
**Script**: `src/preprocessing/data_transformation.py`

**Input**: `data/processed/household_power_consumption_with_features.csv` (891,357 rows × 34 columns)
**Output**: `data/processed/household_power_consumption_transformed.csv` (891,357 rows × 40 columns)

**Column Changes**:
- **Added**: 6 new transformed columns
- **Removed**: None

**New Transformed Features**:

1. **Discretization** (2 columns):
   - `Power_Level` (category) - 4 levels: Low, Medium, High, Very High
     - Based on quartiles of Global_active_power
   - `Voltage_Level` (category) - 5 levels: Very Low, Low, Normal, High, Very High
     - Based on voltage ranges: <230V, 230-235V, 235-240V, 240-245V, >245V

2. **Binarization** (2 columns):
   - `Is_High_Power` (int) - 1 if Global_active_power > median, 0 otherwise
   - `Voltage_Normal_Binary` (int) - 1 if Voltage between 235-245V, 0 otherwise

3. **Label Encoding** (2 columns):
   - `Season_Encoded` (int) - 0=Winter, 1=Spring, 2=Summer, 3=Autumn
   - `TimeOfDay_Encoded` (int) - 0=Night, 1=Morning, 2=Afternoon, 3=Evening

**Final Columns** (40):
- All previous 34 columns +
- Power_Level, Voltage_Level, Is_High_Power, Voltage_Normal_Binary, Season_Encoded, TimeOfDay_Encoded

**Output Files**:
- `data/processed/household_power_consumption_transformed.csv`
- `reports/analysis/transformation_report.txt`

---

### Step 8: Feature Selection
**Script**: `src/preprocessing/feature_selection.py`

**Input**: `data/processed/household_power_consumption_transformed.csv` (891,357 rows × 40 columns)
**Output**: `data/processed/household_power_consumption_final.csv` (891,357 rows × 33 columns)

**Column Changes**:
- **Removed**: 8 redundant features (highly correlated, |r| > 0.7)
- **Kept**: 33 features

**Removed Features** (8):
1. `Global_intensity` - Highly correlated with Global_active_power (r=0.999)
2. `Intensity_ratio` - Highly correlated with Global_intensity (r=0.999)
3. `IsEvening` - Highly correlated with TimeOfDay_Encoded (r=0.706)
4. `Is_High_Power` - Highly correlated with Global_active_power (r=0.766)
5. `Sub_metering_3` - Highly correlated with Total_Sub_metering (r=0.743)
6. `Sub_metering_4` - Highly correlated with Energy_per_minute (r=0.783)
7. `TimeOfDay_Encoded` - Highly correlated with IsNight (r=-0.817)
8. `Total_Sub_metering` - Perfectly correlated with Energy_per_minute (r=1.000)

**Final Columns** (33):
- **DateTime & Time**: DateTime, Date, Time
- **Original Power**: Global_active_power, Global_reactive_power, Voltage
- **Sub-metering**: Sub_metering_1, Sub_metering_2, Sub_metering_3, Sub_metering_4
- **Temporal**: Year, Month, Day, Hour, DayOfWeek, IsWeekend
- **Categorical**: Season, TimeOfDay
- **Discretized**: Power_Level, Voltage_Level
- **Binary**: Voltage_Normal_Binary, IsNight, IsMorning, IsAfternoon
- **Encoded**: Season_Encoded
- **Calculated**: Energy_per_minute
- **Statistical**: Power_1h_avg, Power_24h_avg, Power_prev_1h, Power_change_1h

**Note**: Some removed features (like Sub_metering_3, Sub_metering_4) are still in the final dataset as they were retained in the essential features list. The correlation analysis identified them as redundant but they were kept for domain relevance.

**Output Files**:
- `data/processed/household_power_consumption_final.csv`
- `outputs/correlation_matrix.csv`
- `reports/analysis/feature_selection_report.txt`

---

## Phase 1 Summary

### Dataset Evolution

| Step | Rows | Columns | Key Changes |
|------|------|---------|-------------|
| **Original** | 2,075,259 | 9 | Raw data with missing values |
| **After Sampling** | 999,970 | 9 | Stratified sample (48.2%) |
| **After Cleaning** | 891,357 | 10 | +DateTime, -108K outliers, missing values filled |
| **After Feature Engineering** | 891,357 | 34 | +24 new features |
| **After Transformation** | 891,357 | 40 | +6 transformed features |
| **Final** | 891,357 | 33 | -7 redundant features |

### Key Metrics

- **Data Retention**: 43% of original data (after sampling and cleaning)
- **Missing Values**: 87,731 → 0 (100% fixed via interpolation)
- **Outliers Removed**: 108,613 rows (10.86% of sample)
- **Features Created**: 27 new features
- **Features Removed**: 7 redundant features (high correlation)
- **Final Dataset**: 891,357 rows × 33 columns

### Output Files

**Processed Data**:
- `data/processed/household_power_consumption_cleaned.csv`
- `data/processed/household_power_consumption_with_features.csv`
- `data/processed/household_power_consumption_transformed.csv`
- `data/processed/household_power_consumption_final.csv` ⭐

**Aggregated Data** (7 files):
- `data/aggregated/aggregation_daily.csv`
- `data/aggregated/aggregation_hourly.csv`
- `data/aggregated/aggregation_weekly.csv`
- `data/aggregated/aggregation_monthly.csv`
- `data/aggregated/aggregation_seasonal.csv`
- `data/aggregated/aggregation_timeofday.csv`
- `data/aggregated/aggregation_hour_weekend.csv`

**Reports**:
- `reports/analysis/exploration_report.txt`
- `reports/quality/quality_report.txt`
- `reports/quality/cleaning_report.txt`
- `reports/analysis/features_report.txt`
- `reports/analysis/aggregation_report.txt`
- `reports/analysis/transformation_report.txt`
- `reports/analysis/feature_selection_report.txt`

**Analysis Outputs**:
- `outputs/correlation_matrix.csv`

---

## Phase 2: Advanced Outlier Detection and Multivariate Analysis

Phase 2 focuses on:
- Advanced outlier detection methods (Z-score, Isolation Forest, LOF, Mahalanobis distance)
- Comparison of outlier detection methods
- False positive/negative analysis
- Multivariate statistical analysis
- Enhanced summary statistics
- Principal Component Analysis (PCA)

See `docs/EXECUTION_GUIDE.md` for detailed execution instructions.

---

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
