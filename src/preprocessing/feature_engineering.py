"""
Krijimi i Features të Reja (Feature Engineering)
Kërkesat: Krijimi i vetive
"""

import pandas as pd
import numpy as np

print("="*80)
print("KRIJIMI I FEATURES TË REJA")
print("="*80)

df = pd.read_csv('household_power_consumption_cleaned.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

print(f"\nDataset fillestare: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")
print(f"Periudha: {df['DateTime'].min()} deri {df['DateTime'].max()}")

# Features kohore
print("\n" + "-"*80)
print("FEATURES KOHORE")
print("-"*80)

df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute
df['DayOfWeek'] = df['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['DayName'] = df['DateTime'].dt.day_name()
df['MonthName'] = df['DateTime'].dt.month_name()
df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week

print("✓ Krijuar:")
print("  - Year, Month, Day")
print("  - Hour, Minute")
print("  - DayOfWeek (0-6), DayName")
print("  - MonthName, WeekOfYear")

# Features binary
print("\n" + "-"*80)
print("FEATURES BINARY")
print("-"*80)

df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] < 6)).astype(int)
df['IsMorning'] = ((df['Hour'] >= 6) & (df['Hour'] < 12)).astype(int)
df['IsAfternoon'] = ((df['Hour'] >= 12) & (df['Hour'] < 18)).astype(int)
df['IsEvening'] = ((df['Hour'] >= 18) & (df['Hour'] < 22)).astype(int)

print("✓ Krijuar:")
print(f"  - IsWeekend: {df['IsWeekend'].sum():,} rreshta ({df['IsWeekend'].mean()*100:.1f}%)")
print(f"  - IsNight: {df['IsNight'].sum():,} rreshta ({df['IsNight'].mean()*100:.1f}%)")
print(f"  - IsMorning: {df['IsMorning'].sum():,} rreshta ({df['IsMorning'].mean()*100:.1f}%)")
print(f"  - IsAfternoon: {df['IsAfternoon'].sum():,} rreshta ({df['IsAfternoon'].mean()*100:.1f}%)")
print(f"  - IsEvening: {df['IsEvening'].sum():,} rreshta ({df['IsEvening'].mean()*100:.1f}%)")

# Features kategorike
print("\n" + "-"*80)
print("FEATURES KATEGORIKE")
print("-"*80)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['Season'] = df['Month'].apply(get_season)

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)

print("✓ Season:")
print(df['Season'].value_counts().sort_index())

print("\n✓ TimeOfDay:")
print(df['TimeOfDay'].value_counts())

# Features të kalkuluara
print("\n" + "-"*80)
print("FEATURES TË KALKULUARA")
print("-"*80)

# Sub_metering_4: energia jo e termiket (unmeasured)
df['Sub_metering_4'] = (df['Global_active_power'] * 1000 / 60) - \
                        (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
df['Sub_metering_4'] = df['Sub_metering_4'].clip(lower=0)

print("✓ Sub_metering_4 (energia jo e termiket):")
print(f"  Mean: {df['Sub_metering_4'].mean():.2f} Wh")
print(f"  Min: {df['Sub_metering_4'].min():.2f} Wh")
print(f"  Max: {df['Sub_metering_4'].max():.2f} Wh")

# Total consumption
df['Total_Sub_metering'] = df['Sub_metering_1'] + df['Sub_metering_2'] + \
                            df['Sub_metering_3'] + df['Sub_metering_4']

print("\n✓ Total_Sub_metering:")
print(f"  Mean: {df['Total_Sub_metering'].mean():.2f} Wh")
print(f"  Min: {df['Total_Sub_metering'].min():.2f} Wh")
print(f"  Max: {df['Total_Sub_metering'].max():.2f} Wh")

# Power per hour (kW → kWh për 1 minutë)
df['Energy_per_minute'] = df['Global_active_power'] / 60

print("\n✓ Energy_per_minute (kWh):")
print(f"  Mean: {df['Energy_per_minute'].mean():.4f} kWh")
print(f"  Daily estimate: {df['Energy_per_minute'].mean() * 1440:.2f} kWh")

# Intensity ratio
df['Intensity_ratio'] = df['Global_intensity'] / (df['Voltage'] / 1000)
df['Intensity_ratio'] = df['Intensity_ratio'].replace([np.inf, -np.inf], 0)

print("\n✓ Intensity_ratio (I/V):")
print(f"  Mean: {df['Intensity_ratio'].mean():.4f}")

# Features statistike (rolling averages)
print("\n" + "-"*80)
print("FEATURES STATISTIKE (ROLLING AVERAGES)")
print("-"*80)

print("Duke kalkuluar rolling averages (mund të marrë pak kohë)...")

# 1-hour rolling average (60 minutes)
df['Power_1h_avg'] = df['Global_active_power'].rolling(window=60, min_periods=1).mean()

# 24-hour rolling average (1440 minutes)
df['Power_24h_avg'] = df['Global_active_power'].rolling(window=1440, min_periods=1).mean()

print("✓ Power_1h_avg (mesatare 1 orë):")
print(f"  Mean: {df['Power_1h_avg'].mean():.3f} kW")

print("✓ Power_24h_avg (mesatare 24 orë):")
print(f"  Mean: {df['Power_24h_avg'].mean():.3f} kW")

# Lag features (previous hour)
df['Power_prev_1h'] = df['Global_active_power'].shift(60)
df['Power_change_1h'] = df['Global_active_power'] - df['Power_prev_1h']

print("\n✓ Power_change_1h (ndryshimi nga ora e kaluar):")
print(f"  Mean: {df['Power_change_1h'].mean():.3f} kW")
print(f"  Std: {df['Power_change_1h'].std():.3f} kW")

# Përmbledhje features
print("\n" + "-"*80)
print("PËRMBLEDHJE E FEATURES")
print("-"*80)

original_cols = ['DateTime', 'Date', 'Time', 'Global_active_power', 'Global_reactive_power',
                 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
new_features = [col for col in df.columns if col not in original_cols]

print(f"\nKolona origjinale: {len(original_cols)}")
print(f"Features të reja: {len(new_features)}")
print(f"Total kolona: {len(df.columns)}")

print("\nFeatures të reja:")
for i, feat in enumerate(new_features, 1):
    print(f"  {i:2d}. {feat}")

# Kontrollo për missing values në features të reja
missing_in_new = df[new_features].isnull().sum()
if missing_in_new.sum() > 0:
    print("\n⚠ Missing values në features të reja:")
    for col in new_features:
        if missing_in_new[col] > 0:
            print(f"  {col}: {missing_in_new[col]:,}")
    
    # Fill missing në rolling features
    for col in ['Power_prev_1h', 'Power_change_1h']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print("\n✓ Missing values u plotësuan")
else:
    print("\n✓ Nuk ka missing values në features të reja")

# Ruajtja e dataset-it me features
print("\n" + "-"*80)
print("RUAJTJA E DATASET-IT")
print("-"*80)

df.to_csv('household_power_consumption_with_features.csv', index=False)

print(f"✓ Dataset u ruajt: household_power_consumption_with_features.csv")
print(f"  Rreshta: {df.shape[0]:,}")
print(f"  Kolona: {df.shape[1]} (fillestare: {len(original_cols)}, të reja: {len(new_features)})")
print(f"  Madhësia: ~{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Ruaj një raport features
with open('features_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORTI I KRIJIMIT TË FEATURES\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona\n")
    f.write(f"Kolona fillestare: {len(original_cols)}\n")
    f.write(f"Features të reja: {len(new_features)}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("FEATURES TË REJA\n")
    f.write("-"*80 + "\n\n")
    
    f.write("1. FEATURES KOHORE:\n")
    f.write("   - Year, Month, Day, Hour, Minute\n")
    f.write("   - DayOfWeek, DayName, MonthName\n")
    f.write("   - WeekOfYear\n\n")
    
    f.write("2. FEATURES BINARY:\n")
    f.write("   - IsWeekend (0/1)\n")
    f.write("   - IsNight, IsMorning, IsAfternoon, IsEvening (0/1)\n\n")
    
    f.write("3. FEATURES KATEGORIKE:\n")
    f.write("   - Season (Winter/Spring/Summer/Autumn)\n")
    f.write("   - TimeOfDay (Morning/Afternoon/Evening/Night)\n\n")
    
    f.write("4. FEATURES TË KALKULUARA:\n")
    f.write("   - Sub_metering_4 (energia jo e termiket)\n")
    f.write("   - Total_Sub_metering\n")
    f.write("   - Energy_per_minute\n")
    f.write("   - Intensity_ratio\n\n")
    
    f.write("5. FEATURES STATISTIKE:\n")
    f.write("   - Power_1h_avg (rolling average 1 orë)\n")
    f.write("   - Power_24h_avg (rolling average 24 orë)\n")
    f.write("   - Power_prev_1h (lag feature)\n")
    f.write("   - Power_change_1h (ndryshimi nga ora e kaluar)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("SHPËRNDARJA E VLERAVE\n")
    f.write("-"*80 + "\n\n")
    
    f.write(f"IsWeekend:\n{df['IsWeekend'].value_counts().to_string()}\n\n")
    f.write(f"Season:\n{df['Season'].value_counts().to_string()}\n\n")
    f.write(f"TimeOfDay:\n{df['TimeOfDay'].value_counts().to_string()}\n\n")

print("✓ Raport u ruajt: features_report.txt")

# Shfaq shembull
print("\n" + "-"*80)
print("SHEMBULL I TË DHËNAVE ME FEATURES")
print("-"*80)
print("\n5 rreshta të rastësishëm:")
sample_cols = ['DateTime', 'Global_active_power', 'Hour', 'IsWeekend', 
               'Season', 'TimeOfDay', 'Sub_metering_4', 'Total_Sub_metering']
print(df[sample_cols].sample(5, random_state=42))

print("\n" + "="*80)
print("✓ Feature Engineering përfundoi me sukses!")
print("="*80)
