import pandas as pd
import numpy as np
import os

print("="*80)
print("Creating Stratified Sample")
print("="*80)

print("\nLoading full dataset...")
df_full = pd.read_csv('household_power_consumption.txt', sep=';', na_values=['?'])
df_full['DateTime'] = pd.to_datetime(df_full['Date'] + ' ' + df_full['Time'], 
                                      format='%d/%m/%Y %H:%M:%S')
df_full['Year'] = df_full['DateTime'].dt.year
df_full['Month'] = df_full['DateTime'].dt.month

print(f"Full dataset size: {len(df_full):,} rows")
print(f"Period: {df_full['DateTime'].min()} to {df_full['DateTime'].max()}")

print("\nFull dataset yearly distribution:")
print(df_full['Year'].value_counts().sort_index())

target_sample_size = 1000000
print(f"\nTarget sample size: {target_sample_size:,} rows")

total_rows = len(df_full)
sampling_ratio = target_sample_size / total_rows
print(f"Sampling ratio: {sampling_ratio:.4f} ({sampling_ratio*100:.2f}%)")

print("\nPerforming stratified sampling...")
df_full['Stratum'] = df_full['Year'].astype(str) + '_' + df_full['Month'].astype(str)
strata = df_full.groupby('Stratum')

sampled_dfs = []
for stratum_name, stratum_df in strata:
    n_samples = max(1, int(len(stratum_df) * sampling_ratio))
    sampled = stratum_df.sample(n=n_samples, random_state=42)
    sampled_dfs.append(sampled)

df_stratified = pd.concat(sampled_dfs, ignore_index=True)
df_stratified = df_stratified.sort_values('DateTime').reset_index(drop=True)

print(f"\nStratified sample created: {len(df_stratified):,} rows")
print(f"Actual sampling ratio: {len(df_stratified)/total_rows:.4f} ({len(df_stratified)/total_rows*100:.1f}%)")

print("\nStratified sample yearly distribution:")
print(df_stratified['Year'].value_counts().sort_index())

output_cols = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
               'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
               'Sub_metering_3']
output_file = 'household_power_consumption_sample.txt'
df_stratified[output_cols].to_csv(output_file, sep=';', index=False)

file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

print(f"\n✓ Stratified sample saved: {output_file}")
print(f"  Rows: {len(df_stratified):,}")
print(f"  Size: {file_size_mb:.1f} MB")
print(f"  Coverage: All years (2006-2010) proportionally represented")
print("\n✓ This sample is ready for your analysis!")
print("="*80)
