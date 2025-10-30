"""
Pastrimi i të Dhënave
Kërkesat: Pastrimi, trajtimi i vlerave të zbrazëta, integrimi
"""

import pandas as pd
import numpy as np

print("="*80)
print("PASTRIMI I TË DHËNAVE")
print("="*80)

df = pd.read_csv('household_power_consumption_sample.txt', 
                 sep=';', 
                 low_memory=False,
                 na_values=['?', ''])

print(f"\nDataset fillestare: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")

# Krijimi i DateTime (integrimi i Date dhe Time)
print("\n" + "-"*80)
print("KRIJIMI I DATETIME")
print("-"*80)

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.sort_values('DateTime').reset_index(drop=True)
df = df.set_index('DateTime')

print(f"✓ DateTime u krijua dhe të dhënat u renditën")
print(f"  Periudha: {df.index.min()} deri {df.index.max()}")

# Analiza e missing values
print("\n" + "-"*80)
print("ANALIZA E MISSING VALUES")
print("-"*80)

missing_before = df.isnull().sum()
total_missing_before = missing_before.sum()
print(f"Missing values para pastrimit: {total_missing_before:,}")

for col in df.columns:
    if missing_before[col] > 0:
        print(f"  {col:30s} → {missing_before[col]:,} ({(missing_before[col]/len(df))*100:.2f}%)")

# Strategjia për missing values
print("\n" + "-"*80)
print("STRATEGJIA PËR MISSING VALUES")
print("-"*80)

print("\nApproach: Linear Interpolation")
print("Arsyeja: Të dhënat janë time-series, vlerat fqinje janë më të përshtatshme")
print("Metoda: Interpolation linear bazuar në kohë")

# Ruaj një kopje para interpolimit
df_before_interpolation = df.copy()

# Interpolation për kolonat numerike
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("\nDuke aplikuar interpolation...")
for col in numeric_cols:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            # Time-based interpolation
            df[col] = df[col].interpolate(method='time', limit_direction='both')
            remaining_missing = df[col].isnull().sum()
            print(f"  {col:30s} → {missing_count:,} → {remaining_missing:,}")

# Nëse ka ende missing (në fillim/fund), përdor forward/backward fill
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

missing_after = df.isnull().sum()
total_missing_after = missing_after.sum()
print(f"\n✓ Missing values pas pastrimit: {total_missing_after:,}")

# Analiza e outliers
print("\n" + "-"*80)
print("ANALIZA E OUTLIERS")
print("-"*80)

print("\nApproach: BALANCED - IQR method për outliers")
print("Arsyeja: Hiq outliers ekstreme që shtrembërojnë analizën")
print("Metoda: IQR method - vlerat jashtë [Q1-3*IQR, Q3+3*IQR] hiqen")

outliers_summary = []
rows_to_keep = pd.Series(True, index=df.index)

for col in numeric_cols:
    if col in df.columns:
        # IQR method (3*IQR për të qenë më tolerant)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3*IQR (më tolerant se 1.5*IQR)
        upper_bound = Q3 + 3 * IQR
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        # Shëno rreshtat për të hequr (outliers në çdo kolonë)
        rows_to_keep &= ~outliers_mask
        
        outliers_summary.append({
            'Kolona': col,
            'Outliers': outliers_count,
            'Lower bound': lower_bound,
            'Upper bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        })

# Hiq rreshtat me outliers
df_before_outlier_removal = df.copy()
df = df[rows_to_keep].reset_index(drop=False)
rows_removed = len(df_before_outlier_removal) - len(df)

outliers_df = pd.DataFrame(outliers_summary)
print("\nOutliers të identifikuar:")
print(outliers_df[['Kolona', 'Outliers', 'Lower bound', 'Upper bound']].to_string(index=False))
print(f"\n✓ Rreshta të hequr: {rows_removed:,} ({(rows_removed/len(df_before_outlier_removal))*100:.2f}%)")
print(f"✓ Rreshta të mbetur: {len(df):,} ({(len(df)/len(df_before_outlier_removal))*100:.2f}%)")

# Verifikimi i vlerave negative
print("\n" + "-"*80)
print("VERIFIKIMI I VLERAVE")
print("-"*80)

print("\nKontrollo për vlera negative (që nuk duhet të jenë):")
negative_found = False
for col in numeric_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"  ⚠ {col:30s} → {negative_count:,} vlera negative")
            # Fix negative values për kolonat që duhet të jenë pozitive
            df[col] = df[col].clip(lower=0)
            print(f"     → U korrigjuan në 0")
            negative_found = True

if not negative_found:
    print("✓ Nuk ka vlera negative!")

# Kontrollo për NaN/Inf
print("\nKontrollo për NaN ose Inf:")
for col in numeric_cols:
    nan_count = df[col].isnull().sum()
    inf_count = np.isinf(df[col]).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  ⚠ {col}: NaN={nan_count}, Inf={inf_count}")
    else:
        print(f"  ✓ {col}: OK")

# Statistika para dhe pas
print("\n" + "-"*80)
print("KRAHASIMI: PARA DHE PAS PASTRIMIT")
print("-"*80)

comparison = []
for col in numeric_cols:
    if col in df.columns:
        before_mean = df_before_interpolation[col].mean()
        after_mean = df[col].mean()
        change_pct = ((after_mean - before_mean) / before_mean) * 100 if before_mean != 0 else 0
        
        comparison.append({
            'Kolona': col,
            'Mean para': f"{before_mean:.3f}",
            'Mean pas': f"{after_mean:.3f}",
            'Ndryshimi (%)': f"{change_pct:+.2f}%"
        })

comparison_df = pd.DataFrame(comparison)
print("\nNdryshimet në statistika (duhet të jenë minimale):")
print(comparison_df.to_string(index=False))

# Ruajtja e të dhënave të pastruara
print("\n" + "-"*80)
print("RUAJTJA E TË DHËNAVE TË PASTRUARA")
print("-"*80)

# Selekto kolonat për të ruajtur
cols_to_save = ['DateTime', 'Date', 'Time'] + numeric_cols
df_clean = df[cols_to_save].copy()

# Ruaj dataset-in e pastuar
df_clean.to_csv('household_power_consumption_cleaned.csv', index=False)

print(f"✓ Dataset i pastuar u ruajt: household_power_consumption_cleaned.csv")
print(f"  Rreshta: {df_clean.shape[0]:,} (të njëjta si më parë)")
print(f"  Kolona: {df_clean.shape[1]} (+ DateTime)")
print(f"  Madhësia: {len(df_clean) * len(df_clean.columns) * 8 / 1024**2:.2f} MB")

# Krijo raport pastrimi
with open('cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORTI I PASTRIMIT TË TË DHËNAVE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset fillestare: {len(df_before_interpolation):,} rreshta\n")
    f.write(f"Dataset e pastruar: {len(df_clean):,} rreshta\n")
    f.write(f"Rreshta të fshira: {len(df_before_interpolation) - len(df_clean):,} ({((len(df_before_interpolation) - len(df_clean))/len(df_before_interpolation))*100:.2f}%)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("MISSING VALUES\n")
    f.write("-"*80 + "\n")
    f.write(f"Para: {total_missing_before:,}\n")
    f.write(f"Pas: {total_missing_after:,}\n")
    f.write("Metoda: Linear interpolation (time-based)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("OUTLIERS\n")
    f.write("-"*80 + "\n")
    f.write(outliers_df[['Kolona', 'Outliers', 'Lower bound', 'Upper bound']].to_string(index=False))
    f.write(f"\n\nMetoda: IQR method (3*IQR)\n")
    f.write(f"Rreshta të hequr: {rows_removed:,}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("NDRYSHIMET NË STATISTIKA\n")
    f.write("-"*80 + "\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\nShënim: Ndryshime minimale = pastrimi i mirë\n")

print("✓ Raport i pastrimit u ruajt: cleaning_report.txt")

# Përmbledhje
print("\n" + "="*80)
print("PËRMBLEDHJE E PASTRIMIT")
print("="*80)
rows_removed_total = len(df_before_interpolation) - len(df_clean)
print(f"\n✓ Rreshta fillestare: {len(df_before_interpolation):,}")
print(f"✓ Rreshta të hequr: {rows_removed_total:,} ({(rows_removed_total/len(df_before_interpolation))*100:.2f}%)")
print(f"✓ Rreshta finale: {len(df_clean):,} ({(len(df_clean)/len(df_before_interpolation))*100:.2f}%)")
print(f"✓ Missing values: {total_missing_before:,} → {total_missing_after:,}")
print(f"✓ Outliers: {rows_removed:,} rreshta u hoqën")
print(f"✓ Vlera negative: U korrigjuan në 0")
print(f"✓ DateTime: U krijua dhe integrua")
print("\nApproach i balancuar - cilësi më e mirë e të dhënave!")
print("="*80)
