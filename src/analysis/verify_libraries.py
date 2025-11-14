def verify_imports():
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
        
        import numpy as np
        print(f"✓ numpy {np.__version__}")
        
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
        
        import scipy
        print(f"✓ scipy {scipy.__version__}")
        
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
        
        import seaborn as sns
        print(f"✓ seaborn {sns.__version__}")
        
        from matplotlib_venn import venn3
        print(f"✓ matplotlib-venn installed")
        
        print("\n✓ All libraries verified successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("\nRun: pip install -r ../../requirements.txt")
        return False

if __name__ == "__main__":
    verify_imports()
