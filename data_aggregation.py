"""
Agregimi i të Dhënave
Kërkesat: Agregimi
"""

import pandas as pd
import numpy as np

print("="*80)
print("AGREGIMI I TË DHËNAVE")
print("="*80)

df = pd.read_csv('household_power_consumption_with_features.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date_Only'] = df['DateTime'].dt.date

print(f"\nDataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")
print(f"Periudha: {df['DateTime'].min()} deri {df['DateTime'].max()}")

# Agregim ditor
print("\n" + "-"*80)
print("AGREGIM DITOR")
print("-"*80)

daily_agg = df.groupby('Date_Only').agg({
    'Global_active_power': ['mean', 'sum', 'min', 'max', 'std'],
    'Global_reactive_power': ['mean', 'sum'],
    'Voltage': ['mean', 'min', 'max', 'std'],
    'Global_intensity': ['mean', 'max'],
    'Sub_metering_1': ['sum', 'mean', 'max'],
    'Sub_metering_2': ['sum', 'mean', 'max'],
    'Sub_metering_3': ['sum', 'mean', 'max'],
    'Sub_metering_4': ['sum', 'mean', 'max'],
    'Total_Sub_metering': ['sum', 'mean'],
    'Energy_per_minute': ['sum']
}).reset_index()

daily_agg.columns = ['_'.join(col).strip('_') for col in daily_agg.columns.values]
daily_agg.rename(columns={'Date_Only': 'Date'}, inplace=True)

daily_agg['Daily_Energy_kWh'] = daily_agg['Energy_per_minute_sum']

print(f"✓ Agregim ditor: {len(daily_agg)} ditë")
print(f"\nShembull (5 ditë të para):")
print(daily_agg[['Date', 'Global_active_power_mean', 'Global_active_power_sum', 
                 'Daily_Energy_kWh', 'Voltage_mean']].head())

daily_agg.to_csv('aggregation_daily.csv', index=False)
print(f"\n✓ Ruajtur: aggregation_daily.csv")

# Agregim sipas orës
print("\n" + "-"*80)
print("AGREGIM SIPAS ORËS (HOURLY PATTERNS)")
print("-"*80)

hourly_agg = df.groupby('Hour').agg({
    'Global_active_power': ['mean', 'std', 'min', 'max'],
    'Voltage': ['mean', 'std'],
    'Global_intensity': ['mean', 'max'],
    'Sub_metering_1': ['mean', 'max'],
    'Sub_metering_2': ['mean', 'max'],
    'Sub_metering_3': ['mean', 'max'],
    'Sub_metering_4': ['mean', 'max'],
    'Total_Sub_metering': ['mean']
}).reset_index()

hourly_agg.columns = ['_'.join(col).strip('_') for col in hourly_agg.columns.values]
hourly_agg.rename(columns={'Hour': 'Hour'}, inplace=True)

print(f"✓ Agregim sipas orës: {len(hourly_agg)} orë (0-23)")
print(f"\nPattern konsumi sipas orës:")
print(hourly_agg[['Hour', 'Global_active_power_mean', 'Global_active_power_max']])

# Gjej peak hours
peak_hour = hourly_agg.loc[hourly_agg['Global_active_power_mean'].idxmax(), 'Hour']
low_hour = hourly_agg.loc[hourly_agg['Global_active_power_mean'].idxmin(), 'Hour']
print(f"\n✓ Peak hour: {int(peak_hour)}:00 ({hourly_agg.loc[hourly_agg['Hour'] == peak_hour, 'Global_active_power_mean'].values[0]:.2f} kW)")
print(f"✓ Lowest hour: {int(low_hour)}:00 ({hourly_agg.loc[hourly_agg['Hour'] == low_hour, 'Global_active_power_mean'].values[0]:.2f} kW)")

hourly_agg.to_csv('aggregation_hourly.csv', index=False)
print(f"\n✓ Ruajtur: aggregation_hourly.csv")

# Agregim javor (weekday vs weekend)
print("\n" + "-"*80)
print("AGREGIM JAVOR (WEEKDAY vs WEEKEND)")
print("-"*80)

df['Day_Type'] = df['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})

weekly_agg = df.groupby('Day_Type').agg({
    'Global_active_power': ['mean', 'std', 'min', 'max'],
    'Voltage': ['mean'],
    'Global_intensity': ['mean'],
    'Sub_metering_1': ['mean', 'sum'],
    'Sub_metering_2': ['mean', 'sum'],
    'Sub_metering_3': ['mean', 'sum'],
    'Sub_metering_4': ['mean', 'sum'],
    'Total_Sub_metering': ['mean', 'sum']
}).reset_index()

weekly_agg.columns = ['_'.join(col).strip('_') for col in weekly_agg.columns.values]
weekly_agg.rename(columns={'Day_Type': 'Day_Type'}, inplace=True)

print(f"✓ Agregim javor:")
print(weekly_agg[['Day_Type', 'Global_active_power_mean', 'Global_active_power_std']])

weekday_mean = weekly_agg.loc[weekly_agg['Day_Type'] == 'Weekday', 'Global_active_power_mean'].values[0]
weekend_mean = weekly_agg.loc[weekly_agg['Day_Type'] == 'Weekend', 'Global_active_power_mean'].values[0]
diff_pct = ((weekend_mean - weekday_mean) / weekday_mean) * 100

print(f"\n✓ Weekday mesatar: {weekday_mean:.3f} kW")
print(f"✓ Weekend mesatar: {weekend_mean:.3f} kW")
print(f"✓ Ndryshimi: {diff_pct:+.1f}%")

weekly_agg.to_csv('aggregation_weekly.csv', index=False)
print(f"\n✓ Ruajtur: aggregation_weekly.csv")

# Agregim mujor
print("\n" + "-"*80)
print("AGREGIM MUJOR (MONTHLY TRENDS)")
print("-"*80)

df['Year_Month'] = df['DateTime'].dt.to_period('M')

monthly_agg = df.groupby('Year_Month').agg({
    'Global_active_power': ['mean', 'sum', 'std'],
    'Voltage': ['mean'],
    'Global_intensity': ['mean'],
    'Sub_metering_1': ['sum', 'mean'],
    'Sub_metering_2': ['sum', 'mean'],
    'Sub_metering_3': ['sum', 'mean'],
    'Sub_metering_4': ['sum', 'mean'],
    'Total_Sub_metering': ['sum', 'mean'],
    'Energy_per_minute': ['sum']
}).reset_index()

monthly_agg.columns = ['_'.join(col).strip('_') for col in monthly_agg.columns.values]
monthly_agg.rename(columns={'Year_Month': 'Year_Month'}, inplace=True)
monthly_agg['Year_Month'] = monthly_agg['Year_Month'].astype(str)

print(f"✓ Agregim mujor: {len(monthly_agg)} muaj")
print(f"\nShembull (6 muaj të parë):")
print(monthly_agg[['Year_Month', 'Global_active_power_mean', 'Global_active_power_sum']].head(6))

monthly_agg.to_csv('aggregation_monthly.csv', index=False)
print(f"\n✓ Ruajtur: aggregation_monthly.csv")

# Agregim sezonat
print("\n" + "-"*80)
print("AGREGIM SIPAS SEZONAVE")
print("-"*80)

seasonal_agg = df.groupby('Season').agg({
    'Global_active_power': ['mean', 'std', 'min', 'max'],
    'Voltage': ['mean'],
    'Sub_metering_1': ['mean', 'sum'],
    'Sub_metering_2': ['mean', 'sum'],
    'Sub_metering_3': ['mean', 'sum'],
    'Sub_metering_4': ['mean', 'sum'],
    'Total_Sub_metering': ['mean', 'sum']
}).reset_index()

seasonal_agg.columns = ['_'.join(col).strip('_') for col in seasonal_agg.columns.values]
seasonal_agg.rename(columns={'Season': 'Season'}, inplace=True)

# Rendit sezonat
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
seasonal_agg['Season'] = pd.Categorical(seasonal_agg['Season'], categories=season_order, ordered=True)
seasonal_agg = seasonal_agg.sort_values('Season')

print(f"✓ Agregim sezonat:")
print(seasonal_agg[['Season', 'Global_active_power_mean', 'Global_active_power_std']])

highest_season = seasonal_agg.loc[seasonal_agg['Global_active_power_mean'].idxmax(), 'Season']
lowest_season = seasonal_agg.loc[seasonal_agg['Global_active_power_mean'].idxmin(), 'Season']
print(f"\n✓ Sezona me konsim më të lartë: {highest_season}")
print(f"✓ Sezona me konsim më të ulët: {lowest_season}")

seasonal_agg.to_csv('aggregation_seasonal.csv', index=False)
print(f"\n✓ Ruajtur: aggregation_seasonal.csv")

# Agregim sipas TimeOfDay
print("\n" + "-"*80)
print("AGREGIM SIPAS PJESËS SË DITËS")
print("-"*80)

timeofday_agg = df.groupby('TimeOfDay').agg({
    'Global_active_power': ['mean', 'std', 'max'],
    'Sub_metering_1': ['mean'],
    'Sub_metering_2': ['mean'],
    'Sub_metering_3': ['mean'],
    'Sub_metering_4': ['mean'],
    'Total_Sub_metering': ['mean']
}).reset_index()

timeofday_agg.columns = ['_'.join(col).strip('_') for col in timeofday_agg.columns.values]
timeofday_agg.rename(columns={'TimeOfDay': 'TimeOfDay'}, inplace=True)

# Rendit
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
timeofday_agg['TimeOfDay'] = pd.Categorical(timeofday_agg['TimeOfDay'], categories=time_order, ordered=True)
timeofday_agg = timeofday_agg.sort_values('TimeOfDay')

print(f"✓ Agregim sipas pjesës së ditës:")
print(timeofday_agg[['TimeOfDay', 'Global_active_power_mean', 'Global_active_power_max']])

timeofday_agg.to_csv('aggregation_timeofday.csv', index=False)
print(f"\n✓ Ruajtur: aggregation_timeofday.csv")

# Agregim kombinuar: Hour + IsWeekend
print("\n" + "-"*80)
print("AGREGIM KOMBINUAR (HOUR × WEEKEND)")
print("-"*80)

hour_weekend_agg = df.groupby(['Hour', 'Day_Type']).agg({
    'Global_active_power': ['mean', 'std'],
    'Total_Sub_metering': ['mean']
}).reset_index()

hour_weekend_agg.columns = ['_'.join(col).strip('_') for col in hour_weekend_agg.columns.values]
hour_weekend_agg.rename(columns={'Hour': 'Hour', 'Day_Type': 'Day_Type'}, inplace=True)

print(f"✓ Agregim Hour × Day_Type: {len(hour_weekend_agg)} kombinime")
print(f"\nShembull (8:00-12:00):")
print(hour_weekend_agg[(hour_weekend_agg['Hour'] >= 8) & (hour_weekend_agg['Hour'] <= 12)][['Hour', 'Day_Type', 'Global_active_power_mean']])

hour_weekend_agg.to_csv('aggregation_hour_weekend.csv', index=False)
print(f"\n✓ Ruajtur: aggregation_hour_weekend.csv")

# Raport agregimi
print("\n" + "-"*80)
print("KRIJIMI I RAPORTIT")
print("-"*80)

with open('aggregation_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORTI I AGREGIMIT TË TË DHËNAVE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset origjinal: {df.shape[0]:,} rreshta\n")
    f.write(f"Periudha: {df['DateTime'].min()} - {df['DateTime'].max()}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("AGREGIMET E KRIJUARA\n")
    f.write("-"*80 + "\n\n")
    
    f.write(f"1. DITOR (aggregation_daily.csv)\n")
    f.write(f"   Rreshta: {len(daily_agg)}\n")
    f.write(f"   Periudha: Çdo ditë\n")
    f.write(f"   Statistika: mean, sum, min, max, std\n\n")
    
    f.write(f"2. SIPAS ORËS (aggregation_hourly.csv)\n")
    f.write(f"   Rreshta: {len(hourly_agg)}\n")
    f.write(f"   Pattern: 24 orë (0-23)\n")
    f.write(f"   Peak hour: {int(peak_hour)}:00\n")
    f.write(f"   Lowest hour: {int(low_hour)}:00\n\n")
    
    f.write(f"3. JAVOR (aggregation_weekly.csv)\n")
    f.write(f"   Rreshta: {len(weekly_agg)}\n")
    f.write(f"   Kategori: Weekday vs Weekend\n")
    f.write(f"   Weekday mean: {weekday_mean:.3f} kW\n")
    f.write(f"   Weekend mean: {weekend_mean:.3f} kW\n")
    f.write(f"   Ndryshimi: {diff_pct:+.1f}%\n\n")
    
    f.write(f"4. MUJOR (aggregation_monthly.csv)\n")
    f.write(f"   Rreshta: {len(monthly_agg)}\n")
    f.write(f"   Periudha: {monthly_agg['Year_Month'].min()} - {monthly_agg['Year_Month'].max()}\n\n")
    
    f.write(f"5. SEZONAT (aggregation_seasonal.csv)\n")
    f.write(f"   Rreshta: {len(seasonal_agg)}\n")
    f.write(f"   Kategori: Winter, Spring, Summer, Autumn\n")
    f.write(f"   Highest: {highest_season}\n")
    f.write(f"   Lowest: {lowest_season}\n\n")
    
    f.write(f"6. PJESA E DITËS (aggregation_timeofday.csv)\n")
    f.write(f"   Rreshta: {len(timeofday_agg)}\n")
    f.write(f"   Kategori: Morning, Afternoon, Evening, Night\n\n")
    
    f.write(f"7. HOUR × WEEKEND (aggregation_hour_weekend.csv)\n")
    f.write(f"   Rreshta: {len(hour_weekend_agg)}\n")
    f.write(f"   Kombinime: 24 orë × 2 day types\n\n")
    
    f.write("-"*80 + "\n")
    f.write("INSIGHTS\n")
    f.write("-"*80 + "\n\n")
    f.write(f"Peak consumption hour: {int(peak_hour)}:00\n")
    f.write(f"Lowest consumption hour: {int(low_hour)}:00\n")
    f.write(f"Weekend vs Weekday: {diff_pct:+.1f}%\n")
    f.write(f"Highest season: {highest_season}\n")
    f.write(f"Lowest season: {lowest_season}\n")

print("✓ Raport u ruajt: aggregation_report.txt")

# Përmbledhje
print("\n" + "="*80)
print("PËRMBLEDHJE E AGREGIMIT")
print("="*80)

print(f"\n✓ 7 agregimet u krijuan:")
print(f"  1. Daily: {len(daily_agg)} ditë")
print(f"  2. Hourly: {len(hourly_agg)} orë")
print(f"  3. Weekly: {len(weekly_agg)} kategori")
print(f"  4. Monthly: {len(monthly_agg)} muaj")
print(f"  5. Seasonal: {len(seasonal_agg)} sezona")
print(f"  6. TimeOfDay: {len(timeofday_agg)} kategori")
print(f"  7. Hour×Weekend: {len(hour_weekend_agg)} kombinime")

print("\n✓ Files të ruajtura:")
print("  - aggregation_daily.csv")
print("  - aggregation_hourly.csv")
print("  - aggregation_weekly.csv")
print("  - aggregation_monthly.csv")
print("  - aggregation_seasonal.csv")
print("  - aggregation_timeofday.csv")
print("  - aggregation_hour_weekend.csv")
print("  - aggregation_report.txt")

print("\n" + "="*80)
print("✓ Agregimi përfundoi me sukses!")
print("="*80)
