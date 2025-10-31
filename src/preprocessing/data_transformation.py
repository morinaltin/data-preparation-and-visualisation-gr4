"""
Transformimi i të Dhënave
Kërkesat: Diskretizimi, binarizimi, transformimi
"""

import pandas as pd
import numpy as np

print("="*80)
print("TRANSFORMIMI I TË DHËNAVE")
print("="*80)

df = pd.read_csv('../../data/processed/household_power_consumption_with_features.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

print(f"\nDataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")

# DISKRETIZIMI
print("\n" + "-"*80)
print("DISKRETIZIMI")
print("-"*80)

# Power Level
bins = [0, 
        df['Global_active_power'].quantile(0.25),
        df['Global_active_power'].quantile(0.50),
        df['Global_active_power'].quantile(0.75),
        df['Global_active_power'].max()]
labels = ['Low', 'Medium', 'High', 'Very High']
df['Power_Level'] = pd.cut(df['Global_active_power'], bins=bins, labels=labels, include_lowest=True)

print(f"✓ Power_Level krijuar:")
print(df['Power_Level'].value_counts().sort_index())

# Voltage Level
voltage_bins = [0, 230, 235, 240, 245, 300]
voltage_labels = ['Very Low', 'Low', 'Normal', 'High', 'Very High']
df['Voltage_Level'] = pd.cut(df['Voltage'], bins=voltage_bins, labels=voltage_labels, include_lowest=True)

print(f"\n✓ Voltage_Level krijuar:")
print(df['Voltage_Level'].value_counts().sort_index())

# BINARIZIMI
print("\n" + "-"*80)
print("BINARIZIMI")
print("-"*80)

# High Power (1 if above median, 0 otherwise)
df['Is_High_Power'] = (df['Global_active_power'] > df['Global_active_power'].median()).astype(int)
print(f"✓ Is_High_Power: {df['Is_High_Power'].sum():,} ({df['Is_High_Power'].mean()*100:.1f}%)")

# Voltage Normal (1 if 235-245V, 0 otherwise)
df['Voltage_Normal_Binary'] = ((df['Voltage'] >= 235) & (df['Voltage'] <= 245)).astype(int)
print(f"✓ Voltage_Normal_Binary: {df['Voltage_Normal_Binary'].sum():,} ({df['Voltage_Normal_Binary'].mean()*100:.1f}%)")

# RUAJTJA
# RUAJTJA
print("\n" + "-"*80)
print("RUAJTJA")
print("-"*80)

df.to_csv('../../data/processed/household_power_consumption_transformed.csv', index=False)
print(f"✓ Ruajtur: data/processed/household_power_consumption_transformed.csv")
print(f"  Kolona të reja: 4 (Power_Level, Voltage_Level, Is_High_Power, Voltage_Normal_Binary)")

# RAPORT
with open('../../reports/analysis/transformation_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORTI I TRANSFORMIMIT\n")
    f.write("="*80 + "\n\n")
    
    f.write("DISKRETIZIMI:\n")
    f.write("  - Power_Level (4 kategori: Low/Medium/High/Very High)\n")
    f.write("  - Voltage_Level (5 kategori: Very Low/Low/Normal/High/Very High)\n\n")
    
    f.write(df['Power_Level'].value_counts().sort_index().to_string())
    f.write("\n\n")
    f.write(df['Voltage_Level'].value_counts().sort_index().to_string())
    f.write("\n\n")
    
    f.write("BINARIZIMI:\n")
    f.write("  - Is_High_Power (0/1)\n")
    f.write(f"    1 (High): {df['Is_High_Power'].sum():,} ({df['Is_High_Power'].mean()*100:.1f}%)\n")
    f.write(f"    0 (Low): {(~df['Is_High_Power'].astype(bool)).sum():,}\n\n")
    f.write("  - Voltage_Normal_Binary (0/1)\n")
    f.write(f"    1 (Normal): {df['Voltage_Normal_Binary'].sum():,} ({df['Voltage_Normal_Binary'].mean()*100:.1f}%)\n")
    f.write(f"    0 (Abnormal): {(~df['Voltage_Normal_Binary'].astype(bool)).sum():,}\n")

print("✓ Raport: reports/analysis/transformation_report.txt")

print("\n" + "="*80)
print("✓ Transformimi përfundoi!")
print("="*80)
