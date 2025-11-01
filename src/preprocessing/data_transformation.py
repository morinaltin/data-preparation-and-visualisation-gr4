import pandas as pd
import numpy as np

print("="*80)
print("TRANSFORMIMI I TË DHËNAVE")
print("="*80)

df = pd.read_csv('../../data/processed/household_power_consumption_with_features.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

print(f"\nDataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")

print("\n" + "-"*80)
print("DISKRETIZIMI")
print("-"*80)

bins = [0,
        df['Global_active_power'].quantile(0.25),
        df['Global_active_power'].quantile(0.50),
        df['Global_active_power'].quantile(0.75),
        df['Global_active_power'].max()]
labels = ['Low', 'Medium', 'High', 'Very High']
df['Power_Level'] = pd.cut(df['Global_active_power'], bins=bins, labels=labels, include_lowest=True)

print(f"✓ Power_Level krijuar:")
print(df['Power_Level'].value_counts().sort_index())

voltage_bins = [0, 230, 235, 240, 245, 300]
voltage_labels = ['Very Low', 'Low', 'Normal', 'High', 'Very High']
df['Voltage_Level'] = pd.cut(df['Voltage'], bins=voltage_bins, labels=voltage_labels, include_lowest=True)

print(f"\n✓ Voltage_Level krijuar:")
print(df['Voltage_Level'].value_counts().sort_index())

print("\n" + "-"*80)
print("BINARIZIMI")
print("-"*80)

df['Is_High_Power'] = (df['Global_active_power'] > df['Global_active_power'].median()).astype(int)
print(f"✓ Is_High_Power: {df['Is_High_Power'].sum():,} ({df['Is_High_Power'].mean()*100:.1f}%)")

df['Voltage_Normal_Binary'] = ((df['Voltage'] >= 235) & (df['Voltage'] <= 245)).astype(int)
print(f"✓ Voltage_Normal_Binary: {df['Voltage_Normal_Binary'].sum():,} ({df['Voltage_Normal_Binary'].mean()*100:.1f}%)")

print("\n" + "-"*80)
print("ENCODING KATEGORIK")
print("-"*80)

season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
df['Season_Encoded'] = df['Season'].map(season_mapping)
print(f"✓ Season_Encoded (Winter=0, Spring=1, Summer=2, Autumn=3)")
print(df['Season_Encoded'].value_counts().sort_index())

time_mapping = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
df['TimeOfDay_Encoded'] = df['TimeOfDay'].map(time_mapping)
print(f"\n✓ TimeOfDay_Encoded (Night=0, Morning=1, Afternoon=2, Evening=3)")
print(df['TimeOfDay_Encoded'].value_counts().sort_index())

print("\n" + "-"*80)
print("RUAJTJA")
print("-"*80)

df.to_csv('../../data/processed/household_power_consumption_transformed.csv', index=False)
print(f"✓ Ruajtur: data/processed/household_power_consumption_transformed.csv")
print(f"  Kolona të reja: 6 (diskretizim, binarizim, encoding)")

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
    f.write(f"    0 (Abnormal): {(~df['Voltage_Normal_Binary'].astype(bool)).sum():,}\n\n")
    
    f.write("ENCODING KATEGORIK:\n")
    f.write("  - Season_Encoded (0=Winter, 1=Spring, 2=Summer, 3=Autumn)\n")
    f.write(df['Season_Encoded'].value_counts().sort_index().to_string())
    f.write("\n\n")
    f.write("  - TimeOfDay_Encoded (0=Night, 1=Morning, 2=Afternoon, 3=Evening)\n")
    f.write(df['TimeOfDay_Encoded'].value_counts().sort_index().to_string())

print("✓ Raport: reports/analysis/transformation_report.txt")

print("\n" + "="*80)
print("✓ Transformimi përfundoi!")
print("="*80)
