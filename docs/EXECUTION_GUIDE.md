  # UDHËZUES KOMPLET - EKZEKUTIMI NGA FILLIMI

## 📋 PARAKUSHTET

1. **Python 3.x** instaluar
2. **Librari të nevojshme:**
   ```bash
   pip install pandas numpy
   ```
3. **Dataset origjinal:** `household_power_consumption.txt` (127 MB)

---

## 🚀 EKZEKUTIMI NGA FILLIMI (8 HAPA)

### **HAPI 1: Mostrimi**

**Çfarë bën:** Krijon një sample 1M rreshta (50 MB) nga 2M origjinale

```bash
cd src/preprocessing
python create_stratified_sample.py
```

**Input:** `household_power_consumption.txt` (127 MB)  
**Output:** `data/raw/household_power_consumption_sample.txt` (50 MB)

---

### **HAPI 2: Eksplorimi**

**Çfarë bën:** Analizon strukturën e të dhënave, statistika përshkruese

```bash
python data_exploration.py
```

**Input:** `data/raw/household_power_consumption_sample.txt`  
**Output:** 
- `reports/analysis/exploration_report.txt`
- `reports/analysis/exploration_statistics.csv`
- `reports/analysis/exploration_sample.csv`

---

### **HAPI 3: Analiza e Kualitetit**

**Çfarë bën:** Identifikon missing values, duplikate, outliers

```bash
python data_quality_analysis.py
```

**Input:** `data/raw/household_power_consumption_sample.txt`  
**Output:**
- `reports/quality/quality_report.txt`
- `reports/quality/quality_missing_values.csv`
- `reports/quality/quality_outliers.csv`

**Rezultate:**
- Missing values: 87,731 (0.97%)
- Duplikate: 0
- Outliers: ~256,973 (25.7%)

---

### **HAPI 4: Pastrimi**

**Çfarë bën:** Mbush missing values, hiq outliers, krijon DateTime

```bash
python data_cleaning.py
```

**Input:** `data/raw/household_power_consumption_sample.txt`  
**Output:**
- `data/processed/household_power_consumption_cleaned.csv` (891K rreshta)
- `reports/quality/cleaning_report.txt`

**Veprime:**
- Interpolation për 87,731 missing values
- Hequr 108,613 outliers (10.86%)
- Krijuar kolona DateTime
- Rezultat: 891,357 rreshta (89% të ruajtur)

---

### **HAPI 5: Feature Engineering**

**Çfarë bën:** Krijon 27 features të reja

```bash
python feature_engineering.py
```

**Input:** `data/processed/household_power_consumption_cleaned.csv`  
**Output:**
- `data/processed/household_power_consumption_with_features.csv` (37 kolona)
- `reports/analysis/features_report.txt`

**Features të krijuara:**
- **Temporal:** Year, Month, Day, Hour, DayOfWeek, IsWeekend, Season, TimeOfDay
- **Calculated:** Sub_metering_4, Total_Sub_metering, Energy_per_minute
- **Statistical:** Rolling averages, lag features

---

### **HAPI 6: Agregimi**

**Çfarë bën:** Krijon 7 agregimet e ndryshme

```bash
python data_aggregation.py
```

**Input:** `data/processed/household_power_consumption_with_features.csv`  
**Output:**
- `data/aggregated/aggregation_daily.csv`
- `data/aggregated/aggregation_hourly.csv`
- `data/aggregated/aggregation_weekly.csv`
- `data/aggregated/aggregation_monthly.csv`
- `data/aggregated/aggregation_seasonal.csv`
- `data/aggregated/aggregation_timeofday.csv`
- `data/aggregated/aggregation_hour_weekend.csv`
- `reports/analysis/aggregation_report.txt`

---

### **HAPI 7: Transformimi**

**Çfarë bën:** Diskretizim, binarizim, encoding

```bash
python data_transformation.py
```

**Input:** `data/processed/household_power_consumption_with_features.csv`  
**Output:**
- `data/processed/household_power_consumption_transformed.csv` (+6 kolona)
- `reports/analysis/transformation_report.txt`

**Transformime:**
- **Diskretizim:** Power_Level (4 kategori), Voltage_Level (5 kategori)
- **Binarizim:** Is_High_Power, Voltage_Normal_Binary
- **Encoding:** Season_Encoded (0-3), TimeOfDay_Encoded (0-3)

---

### **HAPI 8: Feature Selection**

**Çfarë bën:** Analizon korrelacionet, hiq features redundante

```bash
python feature_selection.py
```

**Input:** `data/processed/household_power_consumption_transformed.csv`  
**Output:**
- `data/processed/household_power_consumption_final.csv` (dataset final)
- `outputs/correlation_matrix.csv`
- `reports/analysis/feature_selection_report.txt`

**Veprime:**
- Kalkulon matricën e korrelacionit
- Identifikon korrelacione |r| > 0.7
- Hiq features redundante
- Krijon dataset final për analizë

---

## 📊 RRJEDHA E TË DHËNAVE

```
household_power_consumption.txt (2M rreshta, 127 MB)
    ↓
[HAPI 1] Stratified Sampling (50%)
    ↓
household_power_consumption_sample.txt (1M rreshta, 50 MB)
    ↓
[HAPI 2] Exploration (read-only)
[HAPI 3] Quality Analysis (read-only)
    ↓
[HAPI 4] Cleaning (interpolation + outlier removal)
    ↓
household_power_consumption_cleaned.csv (891K rreshta, 10 kolona)
    ↓
[HAPI 5] Feature Engineering (+27 features)
    ↓
household_power_consumption_with_features.csv (891K rreshta, 37 kolona)
    ↓
[HAPI 6] Aggregation (creates 7 views)
    ↓
[HAPI 7] Transformation (+6 features)
    ↓
household_power_consumption_transformed.csv (891K rreshta, 43 kolona)
    ↓
[HAPI 8] Feature Selection (remove redundant)
    ↓
household_power_consumption_final.csv (891K rreshta, ~35-40 kolona)
```

---

## 📁 STRUKTURA E PROJEKTIT

```
individual+household+electric+power+consumption/
│
├── src/preprocessing/          ← Të gjitha script-et
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
│   ├── raw/                    ← Dataset origjinal & sample
│   ├── processed/              ← Cleaned, featured, transformed, final
│   └── aggregated/             ← 7 aggregations
│
├── reports/
│   ├── analysis/               ← Exploration, features, aggregation, etc.
│   └── quality/                ← Quality, cleaning reports
│
├── outputs/                    ← Correlation matrix, figures
├── notebooks/                  ← Jupyter notebooks (opsionale)
└── docs/                       ← README, guides
```

---

## 🎯 OUTPUTS FINALE

### Dataset Files:
1. **household_power_consumption_sample.txt** - Mostra fillestare
2. **household_power_consumption_cleaned.csv** - Të dhëna të pastruara
3. **household_power_consumption_with_features.csv** - Me features
4. **household_power_consumption_transformed.csv** - Transformuar
5. **household_power_consumption_final.csv** - ⭐ DATASET FINAL

### Aggregations (7):
- daily, hourly, weekly, monthly, seasonal, timeofday, hour_weekend

### Reports (8):
- exploration_report.txt
- quality_report.txt
- cleaning_report.txt
- features_report.txt
- aggregation_report.txt
- transformation_report.txt
- feature_selection_report.txt
- + CSV files për detaje

---

## 📈 REZULTATET

| Metrikë | Vlera |
|---------|-------|
| **Dataset origjinal** | 2,075,259 rreshta × 9 kolona (127 MB) |
| **Sample** | 999,970 rreshta (48.2%) |
| **Pas pastrimit** | 891,357 rreshta (89% e sample) |
| **Të dhëna finale** | 891,357 rreshta × ~35-40 kolona |
| **Total të ruajtur** | 43% e dataset-it origjinal |
| **Missing values** | 87,731 → 0 (100% fixed) |
| **Outliers hequr** | 108,613 (10.86%) |
| **Features krijuara** | 27 features të reja |

---

## ⚙️ TROUBLESHOOTING

### Problem 1: "No module named pandas"
```bash
pip install pandas numpy
# or
python -m pip install pandas numpy
```

### Problem 2: FileNotFoundError
- Sigurohu që je në directory e duhur: `src/preprocessing/`
- Kontrollo që files ekzistojnë në `data/raw/` ose `data/processed/`

### Problem 3: Memory Error
- Përdor sample në vend të dataset-it të plotë
- Sample është i mjaftueshëm për analizë

---

## 🔄 EKZEKUTIM I PLOTË (All Steps)

Për të ekzekutuar të gjitha hapat nga fillimi:

```bash
cd src/preprocessing

python create_stratified_sample.py
python data_exploration.py
python data_quality_analysis.py
python data_cleaning.py
python feature_engineering.py
python data_aggregation.py
python data_transformation.py
python feature_selection.py
```

**Koha totale:** ~5-10 minuta (varet nga makina)

---

## ✅ VERIFIKIMI

Pas ekzekutimit, verifiko që files ekzistojnë:

```bash
# Check data files
ls ../../data/processed/
# Should have: cleaned.csv, with_features.csv, transformed.csv, final.csv

# Check reports
ls ../../reports/analysis/
ls ../../reports/quality/

# Check aggregations
ls ../../data/aggregated/
# Should have 7 CSV files

# Check outputs
ls ../../outputs/
# Should have correlation_matrix.csv
```

---

## 📝 PREZANTIMI

Për prezantim, fokusoje në:

1. **Dataset final:** `household_power_consumption_final.csv`
2. **Korrelacioni:** `correlation_matrix.csv`
3. **Agregimet:** Hourly/Daily patterns
4. **Reports:** Exploration, Quality, Feature Selection

---

## 🎓 KËRKESAT E PROFESORIT - STATUS

| # | Kërkesa | Hapi | Status |
|---|---------|------|--------|
| 1 | Mbledhja e të dhënave | 1, 2 | ✅ |
| 2 | Definimi i tipeve | 2 | ✅ |
| 3 | Kualiteti i të dhënave | 3 | ✅ |
| 4 | Mostrimi | 1 | ✅ |
| 5 | Pastrimi | 4 | ✅ |
| 6 | Identifikimi vlerave zbrazëta | 3 | ✅ |
| 7 | Trajtimi vlerave zbrazëta | 4 | ✅ |
| 8 | Integrimi | 4 | ✅ |
| 9 | Agregimi | 6 | ✅ |
| 10 | Krijimi i vetive | 5 | ✅ |
| 11 | Diskretizimi | 7 | ✅ |
| 12 | Binarizimi | 7 | ✅ |
| 13 | Transformimi | 7 | ✅ |
| 14 | Reduktimi dimensionit | 8 | ✅ |
| 15 | Zgjedhja vetive | 8 | ✅ |

**TOTAL: 15/15 (100%)** ✅

---

**Përditësuar:** 31 Tetor 2025  
**Status:** KOMPLETUAR
