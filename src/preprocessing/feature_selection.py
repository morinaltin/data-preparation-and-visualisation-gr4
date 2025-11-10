import pandas as pd
import numpy as np
import os

print("="*80)
print("ZGJEDHJA E FEATURES DHE ANALIZA E KORRELACIONIT")
print("="*80)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../..')

processed_dir = os.path.join(project_root, 'data/processed')
outputs_dir = os.path.join(project_root, 'outputs')
reports_analysis_dir = os.path.join(project_root, 'reports/analysis')
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(reports_analysis_dir, exist_ok=True)

transformed_data_path = os.path.join(processed_dir, 'household_power_consumption_transformed.csv')
df = pd.read_csv(transformed_data_path)
df['DateTime'] = pd.to_datetime(df['DateTime'])

print(f"\nDataset: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")

print("\n" + "-"*80)
print("PËRZGJEDHJA E FEATURES NUMERIKE")
print("-"*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Year', 'Month', 'Day', 'Minute', 'DayOfWeek', 'WeekOfYear']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"Features numerike për analizë: {len(numeric_cols)}")

print("\n" + "-"*80)
print("MATRICA E KORRELACIONIT")
print("-"*80)

correlation_matrix = df[numeric_cols].corr()
print(f"\nMatrica: {correlation_matrix.shape[0]} × {correlation_matrix.shape[1]}")

print("\n" + "-"*80)
print("KORRELACIONE TË FORTA (|r| > 0.7)")
print("-"*80)

high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr.append({
                'Feature_1': correlation_matrix.columns[i],
                'Feature_2': correlation_matrix.columns[j],
                'Correlation': corr_val
            })

if high_corr:
    high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False)
    print(f"\nGjetur {len(high_corr)} korrelacione të forta:")
    print(high_corr_df.to_string(index=False))
else:
    print("\nNuk ka korrelacione shumë të forta (|r| > 0.7)")

print("\n" + "-"*80)
print("FEATURES REDUNDANTE")
print("-"*80)

features_to_remove = set()
for idx, row in enumerate(high_corr):
    feat1 = row['Feature_1']
    feat2 = row['Feature_2']
    
    if 'Global_active_power' not in [feat1, feat2]:
        corr1 = abs(correlation_matrix.loc[feat1, 'Global_active_power'])
        corr2 = abs(correlation_matrix.loc[feat2, 'Global_active_power'])
        
        if corr1 < corr2:
            features_to_remove.add(feat1)
        else:
            features_to_remove.add(feat2)

if features_to_remove:
    print(f"\nFeatures për t'u hequr ({len(features_to_remove)}):")
    for feat in sorted(features_to_remove):
        print(f"  - {feat}")
else:
    print("\nNuk ka features redundante për të hequr")

print("\n" + "-"*80)
print("ZGJEDHJA E FEATURES FINALE")
print("-"*80)

essential_features = [
    'DateTime', 'Date', 'Time',
    'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_4',
    'Year', 'Month', 'Day', 'Hour', 'DayOfWeek',
    'IsWeekend', 'Season', 'TimeOfDay',
    'Power_Level', 'Voltage_Level',
    'Is_High_Power', 'Voltage_Normal_Binary',
    'Season_Encoded', 'TimeOfDay_Encoded'
]

additional_features = [col for col in df.columns
                       if col not in essential_features 
                       and col not in features_to_remove
                       and col in numeric_cols]

selected_features = essential_features + additional_features
selected_features = [col for col in selected_features if col in df.columns]

df_final = df[selected_features].copy()

print(f"\nFeatures fillestare: {df.shape[1]}")
print(f"Features të hequra: {len(features_to_remove)}")
print(f"Features finale: {len(selected_features)}")

print("\n" + "-"*80)
print("STATISTIKA PËR FEATURES FINALE")
print("-"*80)

numeric_selected = df_final.select_dtypes(include=[np.number]).columns
print(f"\nNumerike: {len(numeric_selected)}")
print(f"Kategorike: {len(selected_features) - len(numeric_selected)}")

print("\n" + "-"*80)
print("RUAJTJA E DATASET-IT FINAL")
print("-"*80)

final_data_path = os.path.join(processed_dir, 'household_power_consumption_final.csv')
df_final.to_csv(final_data_path, index=False)
print(f"\n✓ Dataset final u ruajt: data/processed/household_power_consumption_final.csv")
print(f"  Rreshta: {df_final.shape[0]:,}")
print(f"  Kolona: {df_final.shape[1]}")

correlation_matrix_path = os.path.join(outputs_dir, 'correlation_matrix.csv')
correlation_matrix.to_csv(correlation_matrix_path)
print(f"✓ Matrica e korrelacionit: outputs/correlation_matrix.csv")

feature_selection_report_path = os.path.join(reports_analysis_dir, 'feature_selection_report.txt')
with open(feature_selection_report_path, 'w', encoding='utf-8') as f:
    f.write("RAPORTI I ZGJEDHJES SË FEATURES\n")
    f.write("="*80 + "\n\n")
    
    f.write("PËRMBLEDHJE:\n")
    f.write(f"  Features fillestare: {df.shape[1]}\n")
    f.write(f"  Features të hequra: {len(features_to_remove)}\n")
    f.write(f"  Features finale: {len(selected_features)}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("KORRELACIONE TË FORTA (|r| > 0.7)\n")
    f.write("-"*80 + "\n")
    if high_corr:
        f.write(high_corr_df.to_string(index=False))
    else:
        f.write("Nuk ka korrelacione të forta\n")
    f.write("\n\n")
    
    if features_to_remove:
        f.write("-"*80 + "\n")
        f.write("FEATURES TË HEQURA\n")
        f.write("-"*80 + "\n")
        for feat in sorted(features_to_remove):
            f.write(f"  - {feat}\n")
        f.write("\n")
    
    f.write("-"*80 + "\n")
    f.write("FEATURES FINALE\n")
    f.write("-"*80 + "\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")

print(f"✓ Raport: {feature_selection_report_path}")

print("\n" + "="*80)
print("PËRMBLEDHJE E PARA-PROCESIMIT")
print("="*80)

print("\n✓ Dataset origjinal: 2,075,259 rreshta")
print(f"✓ Dataset final: {df_final.shape[0]:,} rreshta ({(df_final.shape[0]/2075259)*100:.1f}%)")
print(f"✓ Features finale: {df_final.shape[1]} kolona")
print(f"✓ Memoria: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n✓ Të gjitha hapat e para-procesimit janë kompletuar!")
print("✓ Dataset final gati për analizë dhe modelim")
print("="*80)
