import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_final_dataset():
    data_path = '../../data/processed/household_power_consumption_cleaned.csv'
    
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        print(f"  ✓ DateTime column parsed")
    
    print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def get_numeric_features(df):
    exclude_cols = [
        'DateTime', 'Date', 'Time',
        'DayName', 'MonthName',
        'Season', 'TimeOfDay',
        'Power_Level', 'Voltage_Level'
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"  ✓ Identified {len(numeric_cols)} numeric features")
    return numeric_cols

def save_report(content, filename):
    report_path = f'../../reports/phase2/{filename}'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Saved report: reports/phase2/{filename}")

def save_csv(df, filename):
    output_path = f'../../outputs/phase2/{filename}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved output: outputs/phase2/{filename}")

def create_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_section_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_subsection(title):
    print(f"\n{title}")
    print("-" * len(title))