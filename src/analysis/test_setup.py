from utils import *

def main():
    print_section_header("PHASE 2 SETUP VERIFICATION")
    
    print("Test 1: Loading dataset...")
    try:
        df = load_final_dataset()
        print("  ✓ Dataset loaded successfully\n")
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}\n")
        return False
    
    print("Test 2: Extracting numeric features...")
    try:
        features = get_numeric_features(df)
        print(f"  ✓ Found {len(features)} numeric features:")
        for i, feat in enumerate(features[:5], 1):
            print(f"    {i}. {feat}")
        if len(features) > 5:
            print(f"    ... and {len(features) - 5} more\n")
    except Exception as e:
        print(f"  ✗ Error getting features: {e}\n")
        return False
    
    print("Test 3: Creating timestamp...")
    try:
        ts = create_timestamp()
        print(f"  ✓ Timestamp: {ts}\n")
    except Exception as e:
        print(f"  ✗ Error creating timestamp: {e}\n")
        return False
    
    print("Test 4: Testing save functions...")
    try:
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        save_csv(test_df, 'test_output.csv')
        
        test_report = f"Test Report\nGenerated: {ts}\n"
        save_report(test_report, 'test_report.txt')
        
        print("  ✓ Save functions working\n")
    except Exception as e:
        print(f"  ✗ Error with save functions: {e}\n")
        return False
    
    print_section_header("✓ ALL TESTS PASSED - SETUP COMPLETE")
    print("\nYou are ready to proceed with Phase 2 analysis!")
    print("Next step: Install required libraries\n")
    
    return True

if __name__ == "__main__":
    main()
