#!/usr/bin/env python3

"""
Fitness Data Analysis Execution Script

This script runs the complete fitness data analysis pipeline, which includes:
- Data loading and preprocessing
- Exploratory data analysis
- Feature engineering
- Model training and evaluation
- Results visualization

Usage:
    python run_analysis.py
"""

import os
import sys
import importlib
import subprocess

# Check if required packages are installed
required_packages = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'joblib'
]

def check_and_install_packages(packages):
    """Check if required packages are installed, install if missing."""
    missing = []
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("All required packages installed!")

# Ensure src directory is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("Checking required packages...")
    check_and_install_packages(required_packages)
    
    # Import main function from main module
    from src.main import main
    
    # Run the main analysis pipeline
    main()