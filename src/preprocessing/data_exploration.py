"""
Eksplorimi dhe Analiza e të Dhënave
Kërkesat: Mbledhja e të dhënave, definimi i tipeve të dhënave
"""

import pandas as pd
import numpy as np

print("="*80)
print("EKSPLORIMI DHE ANALIZA E TË DHËNAVE")
print("="*80)

df = pd.read_csv('household_power_consumption_sample.txt', 
                 sep=';', 
                 low_memory=False,
                 na_values=['?', ''])

print(f"\nDataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")
print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "-"*80)
print("STRUKTURA E DATASET-IT")
print("-"*80)
print(df.info())

print("\n" + "-"*80)
print("TIPET E TË DHËNAVE")
print("-"*80)
for col in df.columns:
    dtype = df[col].dtype
    non_null = df[col].count()
    null_count = df[col].isnull().sum()
    print(f"{col:30s} → {str(dtype):10s} | Vlera: {non_null:>8,} | Zbrazëta: {null_count:>6,}")

print("\n" + "-"*80)
print("SHEMBUJ TË DHËNASH")
print("-"*80)
print("\n10 rreshta të parë:")
print(df.head(10))

print("\n10 rreshta të fundit:")
print(df.tail(10))

print("\n" + "-"*80)
print("STATISTIKA PËRSHKRUESE")
print("-"*80)
print(df.describe())

stats = df.describe().T
stats['range'] = stats['max'] - stats['min']
stats['variance'] = df.select_dtypes(include=[np.number]).var()
print("\nStatistika shtesë:")
print(stats[['mean', 'std', 'min', 'max', 'range', 'variance']])

print("\n" + "-"*80)
print("VLERA UNIKE")
print("-"*80)
for col in df.columns:
    n_unique = df[col].nunique()
    pct_unique = (n_unique / len(df)) * 100
    print(f"{col:30s} → {n_unique:>8,} ({pct_unique:>6.2f}%)")

print("\n" + "-"*80)
print("PERIUDHA KOHORE")
print("-"*80)
print(f"Data unike: {df['Date'].nunique():,}")
print(f"Periudha: {df['Date'].min()} - {df['Date'].max()}")
print(f"Orë unike: {df['Time'].nunique():,}")

print("\n" + "-"*80)
print("RANGET E VLERAVE NUMERIKE")
print("-"*80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"{col:30s} → Min: {df[col].min():>10.3f} | Max: {df[col].max():>10.3f} | Mean: {df[col].mean():>10.3f}")

print("\n" + "="*80)

# Ruajtja e rezultateve
print("\nDUKE RUAJTUR REZULTATET...")

with open('exploration_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORTI I EKSPLORIMIT TË TË DHËNAVE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona\n")
    f.write(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    f.write(f"Periudha: {df['Date'].min()} - {df['Date'].max()}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("TIPET E TË DHËNAVE\n")
    f.write("-"*80 + "\n")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        f.write(f"{col:30s} → {str(dtype):10s} | Vlera: {non_null:>8,} | Zbrazëta: {null_count:>6,}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("STATISTIKA PËRSHKRUESE\n")
    f.write("-"*80 + "\n")
    f.write(df.describe().to_string())

stats_full = df.describe().T
stats_full['range'] = stats_full['max'] - stats_full['min']
stats_full['variance'] = df.select_dtypes(include=[np.number]).var()
stats_full.to_csv('exploration_statistics.csv')

df.head(50).to_csv('exploration_sample.csv', index=False)

print("✓ Rezultatet u ruajtën:")
print("  - exploration_report.txt (raport i plotë)")
print("  - exploration_statistics.csv (statistika)")
print("  - exploration_sample.csv (50 rreshta shembull)")
print("\n" + "="*80)
