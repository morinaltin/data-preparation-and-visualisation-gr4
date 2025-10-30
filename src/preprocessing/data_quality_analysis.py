"""
Analiza e Kualitetit të të Dhënave
Kërkesat: Kualiteti i të dhënave, identifikimi i vlerave të zbrazëta
"""

import pandas as pd
import numpy as np

print("="*80)
print("ANALIZA E KUALITETIT TË TË DHËNAVE")
print("="*80)

df = pd.read_csv('household_power_consumption_sample.txt', 
                 sep=';', 
                 low_memory=False,
                 na_values=['?', ''])

print(f"\nDataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")

# Vlerat e zbrazëta
print("\n" + "-"*80)
print("VLERAT E ZBRAZËTA (MISSING VALUES)")
print("-"*80)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Kolona': missing.index,
    'Nr. Missing': missing.values,
    'Përqindja (%)': missing_pct.values
})

print(missing_df.to_string(index=False))
total_missing = missing.sum()
total_cells = df.shape[0] * df.shape[1]
print(f"\nTotal missing: {total_missing:,} ({(total_missing/total_cells)*100:.2f}% e të gjitha qelizave)")

if total_missing > 0:
    print("\nKolona me missing values:")
    missing_cols = missing_df[missing_df['Nr. Missing'] > 0].sort_values('Nr. Missing', ascending=False)
    print(missing_cols.to_string(index=False))
else:
    print("\n✓ Nuk ka vlera të zbrazëta!")

# Rreshta duplikatë
print("\n" + "-"*80)
print("RRESHTA DUPLIKATË")
print("-"*80)

duplicates = df.duplicated().sum()
duplicate_pct = (duplicates / len(df)) * 100
print(f"Rreshta duplikatë: {duplicates:,} ({duplicate_pct:.2f}%)")

if duplicates > 0:
    print(f"\n⚠ Gjetur {duplicates:,} rreshta duplikatë!")
    print("\nShembuj të duplikatave:")
    print(df[df.duplicated(keep=False)].head(10))
else:
    print("✓ Nuk ka duplikate!")

# Outliers (IQR method)
print("\n" + "-"*80)
print("OUTLIERS (VLERA ANOMALE - IQR METHOD)")
print("-"*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns
outliers_summary = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    n_outliers = outliers_mask.sum()
    outliers_pct = (n_outliers / len(df)) * 100
    
    outliers_summary.append({
        'Kolona': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outliers': n_outliers,
        'Përqindja (%)': outliers_pct,
        'Min': df[col].min(),
        'Max': df[col].max()
    })

outliers_df = pd.DataFrame(outliers_summary)
print("\nPërmbledhje e outliers:")
print(outliers_df[['Kolona', 'Outliers', 'Përqindja (%)', 'Min', 'Max', 'Lower Bound', 'Upper Bound']].to_string(index=False))

# Vlera negative (për kolonat që nuk duhet të jenë negative)
print("\n" + "-"*80)
print("VLERA NEGATIVE (për kolonat që duhet të jenë pozitive)")
print("-"*80)

positive_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

negative_found = False
for col in positive_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"{col:30s} → {negative_count:,} vlera negative")
            negative_found = True

if not negative_found:
    print("✓ Nuk ka vlera negative në kolonat që duhet të jenë pozitive!")

# Vlera zero
print("\n" + "-"*80)
print("VLERA ZERO")
print("-"*80)

for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    zero_pct = (zero_count / len(df)) * 100
    print(f"{col:30s} → {zero_count:>8,} ({zero_pct:>5.2f}%)")

# Konsistenca e të dhënave
print("\n" + "-"*80)
print("KONSISTENCA E TË DHËNAVE")
print("-"*80)

# Kontrollo nëse data është në format të saktë
try:
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    print("✓ Format i Date/Time është i saktë")
    
    # Kontrollo për gaps kohore
    df_sorted = df.sort_values('DateTime')
    time_diffs = df_sorted['DateTime'].diff()
    
    print(f"\nIntervali kohor:")
    print(f"  Mesatar: {time_diffs.mean()}")
    print(f"  Median: {time_diffs.median()}")
    print(f"  Min: {time_diffs.min()}")
    print(f"  Max: {time_diffs.max()}")
    
except Exception as e:
    print(f"⚠ Problem me formatin e Date/Time: {e}")

# Përmbledhje e problemeve
print("\n" + "="*80)
print("PËRMBLEDHJE E PROBLEMEVE TË KUALITETIT")
print("="*80)

problems = []
if total_missing > 0:
    problems.append(f"✗ {total_missing:,} vlera të zbrazëta ({(total_missing/total_cells)*100:.2f}%)")
else:
    problems.append("✓ Nuk ka vlera të zbrazëta")

if duplicates > 0:
    problems.append(f"✗ {duplicates:,} rreshta duplikatë")
else:
    problems.append("✓ Nuk ka duplikate")

total_outliers = outliers_df['Outliers'].sum()
if total_outliers > 0:
    problems.append(f"⚠ {total_outliers:,} outliers të identifikuar")
else:
    problems.append("✓ Nuk ka outliers të dukshëm")

print("\n".join(problems))

# Ruajtja e rezultateve
print("\n" + "-"*80)
print("DUKE RUAJTUR REZULTATET...")
print("-"*80)

with open('quality_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORTI I KUALITETIT TË TË DHËNAVE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona\n\n")
    
    f.write("-"*80 + "\n")
    f.write("VLERAT E ZBRAZËTA\n")
    f.write("-"*80 + "\n")
    f.write(missing_df.to_string(index=False))
    f.write(f"\n\nTotal: {total_missing:,} ({(total_missing/total_cells)*100:.2f}%)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("RRESHTA DUPLIKATË\n")
    f.write("-"*80 + "\n")
    f.write(f"Total: {duplicates:,} ({duplicate_pct:.2f}%)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("OUTLIERS\n")
    f.write("-"*80 + "\n")
    f.write(outliers_df[['Kolona', 'Outliers', 'Përqindja (%)', 'Lower Bound', 'Upper Bound']].to_string(index=False))
    f.write(f"\n\nTotal outliers: {total_outliers:,}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("PËRMBLEDHJE E PROBLEMEVE\n")
    f.write("-"*80 + "\n")
    f.write("\n".join(problems))

missing_df.to_csv('quality_missing_values.csv', index=False)
outliers_df.to_csv('quality_outliers.csv', index=False)

if duplicates > 0:
    df[df.duplicated(keep=False)].to_csv('quality_duplicates.csv', index=False)

print("✓ Rezultatet u ruajtën:")
print("  - quality_report.txt (raport i plotë)")
print("  - quality_missing_values.csv (vlera të zbrazëta)")
print("  - quality_outliers.csv (outliers)")
if duplicates > 0:
    print("  - quality_duplicates.csv (duplikate)")

print("\n" + "="*80)
