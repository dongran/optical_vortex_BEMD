# FDTD Simulation Information

## Simulation Parameters

- **Simulation Type**: FDTD (Finite-Difference Time-Domain)
- **Optical Vortex**: l=1 right-handed optical vortex
- **Frequency**: 84 GHz
- **Wavelength**: ~3.57 mm
- **Grid Resolution**: 560 × 560 spatial points
- **Time Steps**: Originally 2000 steps, reduced to 11 steps (1000-1010)
- **Data Format**: CSV with columns: x, y, z, Ex, Ey, Ez

## Data Files

### Current Data
- **exy1005.csv** (24MB): Single time step (step 1005) used for BEMD example
- **tmp.png** (713KB): Visualization of FDTD simulation field

### Field Components
- **Ex, Ey, Ez**: Electric field components in x, y, z directions
- **Spatial Domain**: 2D grid representing electromagnetic field distribution
- **Physical Units**: Electric field in V/m

## Optical Vortex Characteristics

- **Topological Charge**: l = 1 (single vortex)
- **Handedness**: Right-handed rotation
- **Phase Singularity**: Central null point with phase circulation
- **Intensity Pattern**: Donut-shaped beam profile
- **Vector Field**: Spiral flow pattern around vortex center

## Usage in BEMD Processing

The FDTD data serves as input for:
1. **Data Processing**: Conversion to intensity and vector components
2. **BEMD Decomposition**: Noise separation using empirical mode decomposition
3. **Visualization**: Comparative analysis of original vs processed fields

## File Size Optimization

- **Original Dataset**: ~2GB (2000 time steps)
- **Reduced Dataset**: 24MB (single time step)
- **Reduction Factor**: ~98% size reduction
- **GitHub Compatibility**: Under 25MB file limit 