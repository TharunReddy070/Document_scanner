# Advanced Document Scanner with Step-by-Step Visualization

An interactive, educational document scanning solution that visualizes and explains each step of the document detection and scanning process.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
  - [Edge Detection Method](#edge-detection-method)
  - [Threshold Method](#threshold-method)
  - [Contour Processing](#contour-processing)
  - [Perspective Transformation](#perspective-transformation)
- [Code Structure](#code-structure)
- [Visualization](#visualization)
- [Algorithm Performance](#algorithm-performance)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements an advanced document scanner that detects document boundaries in images and transforms them into clean, flat scans. Unlike conventional scanners, this implementation visualizes each step of the processing pipeline, making it educational and demonstrating how computer vision techniques are applied in document scanning.

The scanner employs two complementary detection methods - edge-based and threshold-based - to maximize document detection success rates across varying lighting and background conditions.

## Features

- **Dual Detection Methods**: Uses both edge-based and threshold-based approaches to maximize document detection success
- **Step-by-Step Visualization**: Shows each processing stage to understand how document detection works
- **Adaptive Parameters**: Automatically tries multiple parameter sets to optimize detection
- **Perspective Transformation**: Corrects document skew and perspective distortion
- **Clean Scan Output**: Produces professional-looking document scans with clean text
- **Comparison View**: Side-by-side comparison of results from different methods
- **Command-line Interface**: Easy to use with customizable options

## Requirements

- Python 3.6+
- OpenCV (cv2) 4.0+
- NumPy
- Pathlib

## Installation

1. Clone or download this repository
2. Install the required dependencies:
```
pip install opencv-python numpy
```

## Usage

Basic usage:
```
python doc_scanner.py path/to/your/image.jpg
```

Advanced options:
```
python doc_scanner.py path/to/your/image.jpg -o output_scan.jpg --no-viz
```

Arguments:
- `image`: Path to the input image containing the document
- `-o, --output`: Path for the output scanned image (default: "scanned_output.jpeg")
- `--no-viz`: Disable visualization windows (useful for batch processing)

## Technical Details

The scanner uses a comprehensive pipeline with two independent methods to detect and scan documents. Each method is explained in detail below:

### Edge Detection Method

1. **Image Preprocessing**:
   - Resizing the image for faster processing while maintaining aspect ratio
   - Converting to grayscale to focus on structural information
   - Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
   - Gaussian blur to reduce noise while preserving edges

2. **Edge Detection**:
   - Auto-Canny algorithm to find optimal threshold values based on image statistics
   - The algorithm calculates lower and upper thresholds using the median pixel intensity
   - Multiple sigma values are tried to optimize edge detection (0.33, 0.25, 0.40)

3. **Edge Enhancement**:
   - Morphological closing to connect nearby edges that might be part of the same boundary
   - Dilation to strengthen and connect edges that might have gaps

4. **Contour Detection**:
   - Finding external contours in the processed edge image
   - Filtering contours based on area (eliminates small noise contours)
   - Approximating contours to find polygonal shapes, focusing on quadrilaterals
   - Applying multiple epsilon values for polygon approximation to find the best document representation

### Threshold Method

1. **Enhanced Preprocessing**:
   - Special border creation around the image to help with documents extending to edges
   - Multiple preprocessing variants are created:
     - Enhanced threshold-specific version with border and contrast adjustment
     - Standard blurred version
     - High contrast version using aggressive CLAHE parameters
     - Normalized version using NORM_MINMAX
     - Sharpened version using a sharpening kernel

2. **Initial Full Document Detection**:
   - Attempts to find the full document using specialized parameters
   - Uses both Otsu's method and adaptive thresholding with carefully chosen parameters

3. **Detailed Threshold Detection**:
   - If full document detection fails, tries multiple threshold techniques:
     - Adaptive thresholding with varying block sizes and C values
     - Otsu's automatic thresholding
     - Simple thresholding with different values

4. **Threshold Refinement**:
   - Multiple morphological operations to clean up the threshold result:
     - Open followed by close
     - Close followed by open
     - Dilate
     - Erode followed by dilate

5. **Multiple Parameter Combinations**:
   - Systematically tries combinations of:
     - Different preprocessing variants
     - Different thresholding methods and parameters
     - Various morphological operations
     - Different epsilon values for contour approximation
     - Both normal and inverted images

### Contour Processing

1. **Contour Evaluation**:
   - Each potential document contour is scored based on:
     - Area (larger is better)
     - Rectangularity (how well it fits a rectangle)
     - Aspect ratio (preference for reasonable document proportions)

2. **Contour Selection**:
   - Ideal case: 4-sided contour with good properties
   - Fallback: 3-6 sided shapes converted to rectangles using minimum area rectangle

### Perspective Transformation

1. **Corner Ordering**:
   - Points are ordered as: top-left, top-right, bottom-right, bottom-left
   - Uses coordinate sums and differences to determine corner positions

2. **Destination Mapping**:
   - Calculates the dimensions of the output document based on the detected corners
   - Creates a "bird's eye view" mapping for the perspective transform

3. **Transformation Application**:
   - Computes the perspective transform matrix using OpenCV's getPerspectiveTransform
   - Applies warpPerspective to obtain the corrected document view

4. **Final Processing**:
   - Converts to grayscale
   - Applies adaptive thresholding for the clean scan effect

## Code Structure

The code is organized into several key functions:

- **Edge Detection Functions**:
  - `auto_canny()`: Implements adaptive Canny edge detection
  - `order_pts()`: Orders points in a consistent manner for perspective transformation
  - `find_document_contours()`: Detects and evaluates document contours with visualization
  - `enhance_for_threshold()`: Special preprocessing for threshold-based detection
  - `four_point_warp()`: Performs perspective transformation with visualization

- **Main Scanning Function**:
  - `scan()`: Orchestrates the entire scanning process with detailed visualization

## Visualization

The scanner provides comprehensive visualization at each step:

1. **Original and Preprocessed Images**:
   - Original image
   - Resized image
   - Grayscale conversion
   - Contrast enhancement
   - Blurred image

2. **Edge Detection Steps**:
   - Canny edges
   - Morphological closing
   - Dilated edges
   - Contour detection with color-coded contours (rejected in red, large in blue, selected in green)
   - Corner detection with labeled points

3. **Threshold Detection Steps**:
   - Various preprocessing variants
   - Threshold results
   - Morphological operations
   - Contour detection with color-coded visualization

4. **Transformation Visualization**:
   - Document corners with color-coded labels (TL, TR, BR, BL)
   - Target rectangle for perspective transformation
   - Warped document result

5. **Result Comparison**:
   - Side-by-side display of edge and threshold methods
   - Final scan results

## Algorithm Performance

The algorithm employs several strategies to maximize document detection success:

1. **Adaptive Parameter Selection**:
   - Automatically tries multiple parameter sets for edge detection and thresholding
   - Falls back to increasingly relaxed parameters if strict ones fail

2. **Multi-method Approach**:
   - Uses both edge and threshold methods independently
   - Each method produces its own result for comparison

3. **Visual Feedback**:
   - Provides visualization at each step to understand algorithm decisions
   - Highlights successful and rejected contours with different colors

## Troubleshooting

Common issues and solutions:

1. **Document Not Detected**:
   - Ensure document has sufficient contrast with background
   - Improve lighting conditions and reduce shadows
   - Try capturing the image with the document fully within frame

2. **Poor Quality Scan**:
   - Ensure the original image is clear and in focus
   - Improve lighting to reduce shadows and glare
   - Hold camera steady when capturing the image

3. **Program Crashes or Freezes**:
   - Try using smaller images (large images may require more memory)
   - Use the `--no-viz` option to disable visualization if memory is limited

4. **Missing Dependencies**:
   - Ensure all required libraries are installed using `pip install opencv-python numpy`
   - Check Python version compatibility (3.6+ recommended)

---

This document scanner provides both practical functionality and educational value, making it perfect for understanding computer vision techniques in document processing applications. 