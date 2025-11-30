# Camera Homography Calibration Tool

A user-friendly GUI application for computing homography transformations between two cameras using checkerboard calibration patterns.

## Overview

This tool helps you calculate the homography matrix that maps points from one camera view to another. It's useful for:
- Multi-camera setups
- Stereo vision applications
- Image alignment and registration
- Augmented reality projects

## Features

### 🎥 Two Operating Modes

1. **Online Mode (Live Cameras)**
   - Real-time camera feed preview
   - Live capture of calibration images
   - Immediate feedback on checkerboard detection
   - Support for multiple USB cameras

2. **Offline Mode (Upload Images)**
   - Upload pre-captured image pairs
   - Process multiple calibration images at once
   - Ideal for when cameras are not directly accessible

### 📊 Comprehensive Results Dashboard

- **Warped Perspective View**: Visual overlay showing alignment quality
- **Matched Corners Visualization**: Side-by-side view of detected checkerboard corners
- **Homography Matrix Display**: Full matrix values for technical review
- **Reprojection Error Analysis**: Automatic quality assessment with color-coded feedback

### ✅ Quality Control

- Automatic checkerboard detection validation
- Mean reprojection error calculation
- Visual and numerical quality indicators
- Warnings for poor calibration results

### 💾 Export Options

- Save homography matrix as NumPy (.npy) file
- Save as text (.txt) file for easy sharing
- Reusable matrix for production applications

## Requirements

### Python Version
- Python 3.6 or higher

### Required Libraries

```bash
pip install opencv-python numpy pillow
```

## Usage
1. Clone the repository:
   ```bash
    git clone https://github.com/MlLearnerAkash/camera_calib.git
    cd camera_calib
    python camera_calib.py
    ```
2. **It will open a Tkniter application**
### References
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Zhang, Z. (2000). A flexible new technique for camera calibration. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 1330-1334.
https://doi.org/10.1109/34.888718
- Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision. Cambridge University Press.
- Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

### Citation to thei repo
If you use this tool in your research, please cite the following:
```
@misc{camera-homography-calibration-tool,
  author = {Akash manna},
  title = {Camera Homography Calibration Tool},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MlLearnerAkash/camera_calib.git}}

}
```