#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing: Convert FDTD Simulation Data to MATLAB Format

This script processes FDTD electromagnetic simulation data:
1. Reads CSV files containing Ex, Ey, Ez field components
2. Calculates field intensity and vector components
3. Crops the data to focus on the optical vortex region
4. Generates visualization plots
5. Saves data in MATLAB format for subsequent BEMD processing

@author: BEMD Optical Vortex Processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
import cv2
import os

# Disable DISPLAY for server environments
os.environ.pop("DISPLAY", None)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Configuration
target_step = 1005  # Process step 1005 as demonstration
crop_size = 560     # Crop to 560x560 to focus on optical vortex region
data_path = '../loam1/exy/'
output_path = './output/'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

print("=== Data Processing (Single Step Demo) ===")
print(f"Processing step {target_step} as demonstration")

# Process single time step
print(f"Processing step {target_step}")
filename = data_path + f'exy{target_step}.csv'
print(f"Reading: {filename}")

try:
    # Read CSV file (skip first 3 header rows)
    data = pd.read_csv(filename, skiprows=3, header=None)
    
    # Extract field components
    # Actual format: X, Y, ex, ey, ez (5 columns)
    x_coords = data.iloc[:, 0].values
    y_coords = data.iloc[:, 1].values
    Ex = data.iloc[:, 2].values
    Ey = data.iloc[:, 3].values
    Ez = data.iloc[:, 4].values
    
    print(f"Data points: {len(x_coords)}")
    
    # Determine grid dimensions
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    nx, ny = len(unique_x), len(unique_y)
    
    print(f"Original grid size: {nx} x {ny}")
    
    # Reshape data to 2D grids
    Ex_2d = Ex.reshape(nx, ny)
    Ey_2d = Ey.reshape(nx, ny)
    Ez_2d = Ez.reshape(nx, ny)
    
    # Calculate field intensity and vector components
    E_intensity = np.sqrt(Ex_2d**2 + Ey_2d**2 + Ez_2d**2)
    
    # Calculate vector components for optical vortex analysis
    V1 = Ex_2d  # Electric field x-component
    V2 = Ey_2d  # Electric field y-component
    V3 = Ez_2d  # Electric field z-component
    
    # Crop data to focus on optical vortex region
    if nx > crop_size or ny > crop_size:
        # Calculate crop indices to center the optical vortex
        start_x = (nx - crop_size) // 2
        start_y = (ny - crop_size) // 2
        
        E_intensity = E_intensity[start_x:start_x+crop_size, start_y:start_y+crop_size]
        V1 = V1[start_x:start_x+crop_size, start_y:start_y+crop_size]
        V2 = V2[start_x:start_x+crop_size, start_y:start_y+crop_size]
        V3 = V3[start_x:start_x+crop_size, start_y:start_y+crop_size]
        
        print(f"After cropping: {crop_size} x {crop_size}")
    
    print("Data processed successfully!")
    print(f"E intensity range: [{np.min(E_intensity):.2e}, {np.max(E_intensity):.2e}]")
    
    # Generate visualization
    print("Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'FDTD Simulation Data - Step {target_step}', fontsize=16)
    
    # E-field intensity
    im1 = axes[0, 0].imshow(E_intensity, cmap='jet', origin='lower')
    axes[0, 0].set_title('E-field Intensity')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Vector field visualization
    step = 12  # Sampling step for vector field arrows
    x_vec = np.arange(0, crop_size, step)
    y_vec = np.arange(0, crop_size, step)
    X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
    
    # Sample vector components
    V1_sampled = V1[::step, ::step]
    V2_sampled = V2[::step, ::step]
    
    axes[0, 1].quiver(X_vec, Y_vec, V1_sampled, V2_sampled, 
                      E_intensity[::step, ::step], cmap='Reds', scale=1e6)
    axes[0, 1].set_title('Vector Field (V1, V2)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    # V3 component
    im3 = axes[1, 0].imshow(V3, cmap='jet', origin='lower')
    axes[1, 0].set_title('V3 Component')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Phase visualization
    phase = np.arctan2(V2, V1)
    im4 = axes[1, 1].imshow(phase, cmap='hsv', origin='lower')
    axes[1, 1].set_title('Phase (arctan2(V2, V1))')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path + f'processed_data_{target_step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saving data in MATLAB format...")
    
    # Prepare data for MATLAB (add time dimension)
    # Format: (time_steps, height, width) - single time step
    dataE = E_intensity.reshape(1, E_intensity.shape[0], E_intensity.shape[1])
    dataV1 = V1.reshape(1, V1.shape[0], V1.shape[1])
    dataV2 = V2.reshape(1, V2.shape[0], V2.shape[1])
    dataV3 = V3.reshape(1, V3.shape[0], V3.shape[1])
    
    # Save as MATLAB files
    savemat(output_path + f'loam1E{target_step}.mat', {'dataE': dataE})
    savemat(output_path + f'loam1V1{target_step}.mat', {'dataV1': dataV1})
    savemat(output_path + f'loam1V2{target_step}.mat', {'dataV2': dataV2})
    savemat(output_path + f'loam1V3{target_step}.mat', {'dataV3': dataV3})
    
    print("=== Step 1 Complete ===")
    print(f"Processed single time step: {target_step}")
    print(f"Data shape: {dataE.shape}")
    print(f"Output files saved to: {output_path}")
    print("Files generated:")
    print(f"- loam1E{target_step}.mat: Electric field intensity")
    print(f"- loam1V1{target_step}.mat: V1 component")
    print(f"- loam1V2{target_step}.mat: V2 component")
    print(f"- loam1V3{target_step}.mat: V3 component")
    print(f"- processed_data_{target_step}.png: Visualization")
    print("")
    print("Next: Run MATLAB BEMD processing (step2_bemd_processing.m)")
    
except Exception as e:
    print(f"Error processing step {target_step}: {e}")
    print("Please check the input data format and file paths.") 