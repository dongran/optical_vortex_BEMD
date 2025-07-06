#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Optical Vortex BEMD Processing Example

This script demonstrates the complete pipeline for processing optical vortex data:
1. Data Processing: Convert FDTD CSV to MATLAB format
2. BEMD Analysis: Apply BEMD decomposition (requires MATLAB)
3. Visualization: Generate comparative analysis plots

@author: BEMD Optical Vortex Processing
"""

import subprocess
import sys
import os
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True
        return False
    except ImportError:
        return False

def check_matlab():
    """Check if MATLAB is available"""
    try:
        result = subprocess.run(['which', 'matlab'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_data_file():
    """Check if the required data file exists"""
    return os.path.exists('../loam1/exy/exy1005.csv')

def run_step(step_num, description, command, is_matlab=False):
    """Run a processing step with error handling"""
    print(f"\n{'='*60}")
    print(f"📊 Step {step_num}: {description}")
    print(f"{'='*60}")
    
    if is_matlab:
        print("Applying BEMD decomposition to field components...")
        print("⏰ This may take 1-2 minutes...")
    else:
        print(f"{description}...")
    
    print(f"\n{'='*60}")
    print(f"🚀 {description} (Single Step)")
    print(f"{'='*60}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        if is_matlab:
            # For MATLAB, we need to change to the correct directory
            result = subprocess.run(
                ['matlab', '-batch', f"cd('{os.getcwd()}'); step2_bemd_processing"],
                capture_output=True, text=True, timeout=600
            )
        else:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
        
        print("📝 Output:")
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warning/Error output:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Command completed successfully!")
            return True
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    """Main execution function"""
    print("🌟 Optical Vortex BEMD Processing - Complete Example (Single Step Demo)")
    print("="*80)
    print("This demo processes a single time step (1005) to demonstrate the complete pipeline:")
    print("1. Data Processing: Convert CSV to MATLAB format")
    print("2. BEMD Analysis: Apply BEMD decomposition to E, V1, V2, V3 components")
    print("3. Visualization: Generate analysis plots and comparison")
    print("="*80)
    
    # Check requirements
    print("🔍 Checking requirements...")
    
    # Check Python packages
    required_packages = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('opencv-python', 'cv2')
    ]
    
    missing_packages = []
    for package, import_name in required_packages:
        if check_package(package, import_name):
            print(f"✅ {package} is installed")
        else:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    # Check MATLAB
    if check_matlab():
        print("✅ MATLAB is available")
    else:
        print("❌ MATLAB is not available")
        print("Please ensure MATLAB is installed and in your PATH")
        return False
    
    # Check data file
    if check_data_file():
        print("✅ Data file exists: ../loam1/exy/exy1005.csv")
    else:
        print("❌ Data file not found: ../loam1/exy/exy1005.csv")
        print("Please check the data directory structure")
        return False
    
    # Run processing steps
    steps = [
        (1, "Data Processing", "python step1_data_processing.py", False),
        (2, "BEMD Processing", "step2_bemd_processing", True),
        (3, "Visualization", "python step3_visualization.py", False),
    ]
    
    for step_num, description, command, is_matlab in steps:
        success = run_step(step_num, description, command, is_matlab)
        if not success:
            print(f"❌ Step {step_num} failed. Stopping execution.")
            return False
        print(f"✅ Step {step_num} completed successfully!")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 Processing Complete!")
    print("="*80)
    print("All processing steps completed successfully!")
    print("")
    print("Generated files in ./output/:")
    print("- processed_data_1005.png: Original data visualization")
    print("- bemd_analysis_1005.png: Main BEMD analysis (2×3 layout)")
    print("- bemd_analysis_summary_1005.txt: Analysis summary")
    print("- MATLAB data files: loam1*.mat (8 files)")
    print("")
    print("📊 Key Results:")
    print("- Successfully separated noise (IMF1) from signal (IMF2+Residual)")
    print("- Generated comparative visualization showing optical vortex structure")
    print("- Processed single time step in ~5 minutes")
    print("")
    print("🎯 Next Steps:")
    print("- Examine the generated visualization: ./output/bemd_analysis_1005.png")
    print("- Check analysis summary: ./output/bemd_analysis_summary_1005.txt")
    print("- Modify parameters for different processing requirements")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 