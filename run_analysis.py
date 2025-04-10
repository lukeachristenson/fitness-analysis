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

# Ensure src directory is in the path
sys.path.append(os.path.abspath('src'))

# Import main function from main module
from src.main import main

if __name__ == "__main__":
    # Run the main analysis pipeline
    main()
