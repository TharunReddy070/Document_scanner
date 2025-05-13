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

Edge detection is a fundamental technique in computer vision used to identify boundaries within an image where brightness changes sharply. In the context of document scanning, edge detection helps identify the boundaries of a document against its background.

#### Why Edge Detection is Needed
- Documents typically have well-defined borders that create distinctive edges in an image
- Edges are relatively invariant to lighting conditions (though affected by shadows)
- Edge detection focuses on structural information rather than color or intensity
- Helps isolate the document from potentially complex backgrounds

#### The Complete Edge Detection Pipeline

1. **Image Preprocessing**:
   - **Resizing**: The image is resized while maintaining aspect ratio to reduce computational load. Smaller images process faster while preserving essential structural information.
   - **Grayscale Conversion**: Color information is discarded by converting to grayscale because edges are defined by intensity changes, not color differences. This simplifies processing and focuses on structural information.
   - **Contrast Enhancement (CLAHE)**: Contrast Limited Adaptive Histogram Equalization improves local contrast in different regions of the image. This helps reveal edges that might be faint in the original image. Unlike global histogram equalization, CLAHE operates on small regions (tiles), making it more effective for documents with varying lighting.
   - **Gaussian Blur**: A Gaussian filter is applied to reduce noise while preserving edges. This is critical because noise can be falsely detected as edges. The Gaussian blur smooths the image while respecting edge boundaries due to its weighting function that gives more importance to nearby pixels.

2. **Edge Detection Algorithm**:
   - **Auto-Canny Implementation**: Rather than using fixed thresholds for edge detection, the algorithm calculates optimal thresholds based on the image's statistical properties:
     ```python
     v = np.median(img)
     lower = int(max(0, (1. - sigma) * v))
     upper = int(min(255, (1. + sigma) * v))
     ```
   - **How Canny Works**: The Canny algorithm is a multi-stage process:
     1. Noise reduction using Gaussian filter
     2. Gradient calculation using Sobel filters in both x and y directions
     3. Non-maximum suppression to thin edges
     4. Hysteresis thresholding with upper and lower thresholds to determine strong and weak edges
     5. Edge tracking by hysteresis to include weak edges connected to strong edges
   
   - **Adaptive Sigma Values**: The algorithm tries multiple sigma values (0.33, 0.25, 0.40) to optimize edge detection. These parameters control the width of the threshold window around the median pixel value. Smaller sigma values create tighter thresholds (more selective), while larger values create wider thresholds (more inclusive).

3. **Edge Enhancement**:
   - **Morphological Closing**: This operation (dilation followed by erosion) helps connect nearby edges that might be part of the same boundary but appear disconnected due to lighting variations, texture, or noise. The mathematical expression is: Closing(A,B) = Erosion(Dilation(A,B),B) where A is the image and B is the structuring element.
     ```
     Closing fills small gaps between edge segments, which is crucial for document boundary detection.
     ```
   
   - **Dilation**: This morphological operation expands the white regions in a binary image, helping to connect broken edges. Each pixel in the input image is replaced by the maximum value within its neighborhood as defined by a kernel:
     ```
     Dilation(A,B) = {z | (B̂)z ∩ A ≠ ∅}
     ```
     Where B̂ is the reflection of B and ∩ represents intersection.
   
   - **Kernel Size Selection**: The code tries different kernel sizes (5×5, 7×7) for morphological operations. Larger kernels can bridge wider gaps but might merge unrelated edges.

4. **Contour Analysis for Document Detection**:
   - **Contour Extraction**: After edge processing, the algorithm extracts contours from the resulting binary image using `cv2.findContours()` with `RETR_EXTERNAL` mode to find only the outermost contours.
   - **Contour Filtering**: Contours are filtered based on area, removing small noise contours using a relative area threshold (ratio to total image area).
   - **Polygon Approximation**: The algorithm uses `cv2.approxPolyDP()` to approximate contours with polygons, trying various epsilon values (0.02, 0.01, 0.03, 0.05) of the contour perimeter. This converts curved contours into simpler polygonal shapes.
   - **Quadrilateral Focus**: Particular attention is paid to quadrilateral shapes, as documents are typically rectangular. The closer a contour is to having 4 vertices, the better candidate it is for a document.

### Threshold Method

Thresholding is a segmentation technique that separates objects from their background by converting a grayscale image to binary based on pixel intensity values. For document scanning, this helps isolate the document from its surroundings based on brightness differences.

#### Why Thresholding is Needed
- Edge detection may fail if document edges are not distinct enough
- Thresholding can capture the entire document as a solid region rather than just its edges
- Works well when there's good contrast between the document and background
- Can help identify documents with faint or blurred edges

#### The Complete Threshold Detection Pipeline

1. **Enhanced Preprocessing for Thresholding**:
   - **Border Creation**: A 20-pixel border is added around the image using `cv2.copyMakeBorder()`. This helps detect documents that extend to the edges of the image by creating contrast at the borders:
     ```python
     bordered = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
     ```
   
   - **Multiple Preprocessing Variants**: The algorithm creates several differently processed versions of the image to increase the chances of successful document detection:
     
     a. **Enhanced Threshold-Specific Version**:
     ```python
     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
     contrast = clahe.apply(bordered)
     bilateral = cv2.bilateralFilter(contrast, 9, 75, 75)
     ```
     The bilateral filter is particularly important as it smooths the image while preserving edges, using both spatial and intensity differences to determine weights.
     
     b. **High Contrast Version** using aggressive CLAHE parameters (clipLimit=4.0)
     
     c. **Normalized Version** using `cv2.normalize()` with NORM_MINMAX to stretch the histogram to full dynamic range [0,255]
     
     d. **Sharpened Version** using a kernel:
     ```python
     kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
     sharpened = cv2.filter2D(blurred, -1, kernel)
     ```
     This kernel enhances edges by accentuating the difference between a pixel and its neighbors.

2. **Threshold Application Techniques**:
   
   a. **Otsu's Method**: An automatic thresholding technique that calculates the optimal threshold value by minimizing the intra-class variance between the background and foreground pixels:
   ```python
   _, thresh = cv2.threshold(preproc_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   ```
   The mathematical formula for Otsu's method involves:
   ```
   Finding t that maximizes σ²ᵦ(t) = ω₁(t)ω₂(t)[μ₁(t)-μ₂(t)]²
   ```
   where ω₁, ω₂ are the probabilities of the two classes and μ₁, μ₂ are their means.
   
   b. **Adaptive Thresholding**: Unlike global thresholding, this calculates different threshold values for different regions in the image:
   ```python
   thresh = cv2.adaptiveThreshold(preproc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, block_size, C)
   ```
   - The block_size parameter (11, 21, 31, 15, 25) determines the size of the neighborhood area
   - The C parameter (5, 10, 15, 3, 8) is a constant subtracted from the calculated mean or weighted mean
   - ADAPTIVE_THRESH_GAUSSIAN_C uses a Gaussian-weighted sum to calculate the local threshold
   
   c. **Simple Thresholding**: Fixed threshold values (127, 150, 100) are tried as fallbacks:
   ```python
   _, thresh = cv2.threshold(preproc_img, value, 255, cv2.THRESH_BINARY_INV)
   ```

3. **Threshold Refinement with Morphological Operations**:
   
   After thresholding, the binary image may contain noise, holes, and other irregularities. These are cleaned up using morphological operations:
   
   a. **Open+Close**: Opening (erosion followed by dilation) removes small noise objects, then closing fills holes and connects nearby objects:
   ```python
   cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1), 
                  cv2.MORPH_CLOSE, kernel, iterations=2)
   ```
   
   b. **Close+Open**: The reverse sequence can produce different results depending on image characteristics
   
   c. **Dilate**: Simple expansion of white regions to strengthen potential document shapes
   
   d. **Erode+Dilate**: First shrinking objects to remove thin connections and noise, then expanding to restore size but with cleaner shapes

4. **Systematic Parameter Search**:
   
   The algorithm systematically tries multiple combinations of:
   - Different preprocessing variants
   - Different thresholding methods and parameters
   - Various morphological operations
   - Different epsilon values for contour approximation
   - Both normal and inverted images
   
   This exhaustive search increases the chances of finding the document under varying conditions, essentially creating a robust parameter space exploration.

### Contour Processing

#### What are Contours?
Contours are curves joining all continuous points along a boundary with the same color or intensity. In the context of document scanning, contours represent the outlines or boundaries of objects in the image. 

In mathematical terms, contours can be represented as a sequence of points:
```
C = {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
```

#### Why Contour Processing is Needed
- Contours allow for the identification of document boundaries from binary images
- They provide a compact representation of shapes within the image
- Contours can be analyzed for their geometric properties (area, perimeter, shape)
- They enable the extraction of the specific document region from the background

#### Detailed Contour Processing Pipeline

1. **Contour Extraction**:
   - After edge detection or thresholding, we have a binary image where white pixels represent potential document edges or regions
   - OpenCV's `cv2.findContours()` function is used to extract these contours:
   ```python
   contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```
   - `RETR_EXTERNAL` retrieves only the outermost contours, which is appropriate for document detection
   - `CHAIN_APPROX_SIMPLE` compresses horizontal, vertical, and diagonal segments, keeping only their end points, reducing memory usage

2. **Contour Filtering and Sorting**:
   - Contours are first sorted by area (largest first) as documents typically form large contours:
   ```python
   contours = sorted(contours, key=cv2.contourArea, reverse=True)
   ```
   - Small contours that are likely noise are filtered out using a minimum area threshold based on the image size:
   ```python
   min_area = img_area * min_area_ratio
   if area < min_area:
       continue
   ```
   - This relative thresholding (using a ratio rather than absolute values) ensures the algorithm works on images of any size

3. **Contour Evaluation and Scoring**:
   - Each potential document contour is evaluated using multiple criteria:
   
   a. **Area**: Larger contours are preferred as they likely represent the document
   
   b. **Rectangularity**: How well the contour fits a rectangle, calculated as:
   ```python
   rect = cv2.minAreaRect(c)
   box = cv2.boxPoints(rect)
   box_area = cv2.contourArea(box)
   rect_ratio = area / box_area if box_area > 0 else 0
   ```
   A perfect rectangle would have rect_ratio = 1.0
   
   c. **Aspect Ratio**: The ratio of the longest side to the shortest side of the bounding rectangle:
   ```python
   rect = cv2.boundingRect(approx)
   aspect_ratio = max(rect[2], rect[3]) / min(rect[2], rect[3])
   ```
   Extreme aspect ratios (very long/thin shapes) are penalized
   
   d. **Combined Scoring**: These metrics are combined into a single score:
   ```python
   contour_score = area * rect_ratio * (1.0 / max(1.0, aspect_ratio))
   ```
   This formula favors large, rectangular contours with reasonable aspect ratios

4. **Polygon Approximation for Shape Analysis**:
   - Document contours are usually polygons with a small number of vertices
   - The Douglas-Peucker algorithm is used to approximate the contour with a simpler polygon:
   ```python
   perimeter = cv2.arcLength(c, True)
   approx = cv2.approxPolyDP(c, factor * perimeter, True)
   ```
   - The epsilon factor controls how closely the approximation matches the original contour
   - Multiple epsilon values (0.02, 0.01, 0.03, 0.05) are tried to find the best approximation

5. **Handling Different Vertex Counts**:
   
   a. **Ideal Case (4 vertices)**:
   - When the approximated polygon has exactly 4 vertices, it's directly used as a document contour
   - These vertices are ordered and used for perspective transformation
   
   b. **Non-Ideal Cases (3, 5, or 6 vertices)**:
   - When the approximated polygon has 3, 5, or 6 vertices, it's converted to a rectangle
   - This is done using the minimum-area rotated rectangle:
   ```python
   rect = cv2.minAreaRect(approx)
   box = cv2.boxPoints(rect)
   box = np.int32(box)
   ```
   - The resulting 4-point box is used as the document contour

6. **Best Contour Selection**:
   - All potential document contours are sorted by their scores
   - The highest-scoring contour is selected as the best representation of the document
   - This contour's points are then passed to the perspective transformation stage

### Perspective Transformation

Perspective transformation corrects the distorted view of a document in an image, creating a flat, frontal view as if the document were scanned by a traditional scanner. This is a crucial step in document scanning.

#### Why Perspective Transformation is Needed
- Documents captured by cameras typically exhibit perspective distortion
- The document might be photographed at an angle, making it appear skewed
- Text and content should be perpendicular to the viewing angle for optimal readability
- Creates a standardized, scanner-like output regardless of how the photo was taken

#### Complete Perspective Transformation Pipeline

1. **Corner Point Ordering**:
   - The 4 points detected from contour processing need to be in a consistent order before transformation
   - The `order_pts()` function reorders points as: top-left, top-right, bottom-right, bottom-left:
   ```python
   def order_pts(pts):
       rect = np.zeros((4, 2), dtype="float32")
       
       # Sum of coordinates - smallest is top-left, largest is bottom-right
       s = pts.sum(axis=1)
       rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
       rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
       
       # Difference of coordinates - smallest is top-right, largest is bottom-left
       diff = np.diff(pts, axis=1).flatten()
       rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
       rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
       
       return rect
   ```
   - This ordering leverages mathematical properties of coordinates in a rectangle:
     - The top-left corner has the smallest sum of x+y coordinates
     - The bottom-right corner has the largest sum of x+y coordinates
     - The top-right corner has the smallest difference of x-y coordinates
     - The bottom-left corner has the largest difference of x-y coordinates

2. **Destination Size Calculation**:
   - The dimensions of the output document are calculated based on the detected corners:
   ```python
   # Compute width: max distance between br/bl or tr/tl
   widthA = np.linalg.norm(br - bl)
   widthB = np.linalg.norm(tr - tl)
   maxWidth = max(1, int(max(widthA, widthB)))

   # Compute height: max distance between tr/br or tl/bl
   heightA = np.linalg.norm(tr - br)
   heightB = np.linalg.norm(tl - bl)
   maxHeight = max(1, int(max(heightA, heightB)))
   ```
   - The Euclidean distances between corners determine the dimensions of the output document
   - Using the maximum of the two possible width and height measurements ensures the output has appropriate dimensions

3. **Perspective Transform Matrix Calculation**:
   - The transformation is calculated using the source points (ordered corners) and destination points:
   ```python
   # Destination points for "bird's eye view"
   dst = np.array([
       [0, 0],                  # Top-left
       [maxWidth - 1, 0],       # Top-right
       [maxWidth - 1, maxHeight - 1], # Bottom-right
       [0, maxHeight - 1]       # Bottom-left
   ], dtype="float32")
   
   M = cv2.getPerspectiveTransform(rect, dst)
   ```
   - The perspective transform matrix M is a 3×3 matrix that maps source points to destination points
   - The mathematical representation is:
   ```
   [x']   [m00 m01 m02]   [x]
   [y'] = [m10 m11 m12] × [y]
   [1 ]   [m20 m21 m22]   [1]
   ```
   where (x,y) are source coordinates and (x',y') are destination coordinates

4. **Perspective Warp Application**:
   - The transform matrix is applied to the entire image:
   ```python
   warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
   ```
   - This creates a new image where the document appears as if viewed directly from above
   - The transformation maintains straight lines but changes their angles to correct perspective

5. **Final Image Processing**:
   - The warped document is converted to grayscale:
   ```python
   warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
   ```
   
   - Adaptive thresholding is applied to create a clean, scanner-like output:
   ```python
   scan_result = cv2.adaptiveThreshold(warped_gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 21, 10)
   ```
   - This binary image enhances text readability and removes shadows, mimicking the output of a traditional scanner

6. **Visualization of Transformation**:
   - The code includes extensive visualization of the transformation process:
     - Color-coded corners (red, green, blue, yellow) with labels (TL, TR, BR, BL)
     - The target rectangle showing where the corners will map to
     - Before and after images showing the original document and its transformed version

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