#!/usr/bin/env python3
"""
Document Scanner with comprehensive visualization of edge and threshold detection.
Shows each processing step to help understand how edges are detected.
"""

import cv2
import sys
import argparse
import numpy as np
from pathlib import Path

# ────────────────────────── Edge detection functions ──────────────────────────

def auto_canny(img, sigma):
    """Applies Canny edge detection using the median pixel intensity."""
    v = np.median(img)
    lower = int(max(0, (1. - sigma) * v))
    upper = int(min(255, (1. + sigma) * v))
    return cv2.Canny(img, lower, upper)

def order_pts(pts):
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    # Ensure pts is a NumPy array with shape (4, 2)
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts, dtype="float32")
    
    # Attempt to reshape if needed
    if pts.shape != (4, 2):
        try:
            pts = pts.reshape((4, 2))
        except ValueError:
            print(f"Warning: order_pts received points with unexpected shape: {pts.shape}.")
            # Create default rectangle
            h, w = 100, 100
            pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    
    # Calculate the ordering of points
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

def find_document_contours(img_processed, min_area_ratio=0.1, epsilon_factor=0.02):
    """
    Finds document contours in a processed image, with visualization.
    Returns the contours and a visualization image.
    """
    # Create a visualization image
    viz_img = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
    
    # Find contours
    contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours in light gray
    cv2.drawContours(viz_img, contours, -1, (100, 100, 100), 1)
    
    if not contours:
        return None, viz_img

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Calculate minimum area threshold
    img_area = img_processed.shape[0] * img_processed.shape[1]
    min_area = img_area * min_area_ratio
    
    # Filter and process contours
    possible_docs = []
    
    # Try to find contours that cover most of the image (likely the full document)
    # Consider the largest contours first
    for i, c in enumerate(contours[:5]):  # Only check the 5 largest contours
        area = cv2.contourArea(c)
        if area < min_area:
            # Mark small contours in red
            cv2.drawContours(viz_img, [c], 0, (0, 0, 200), 1)
            continue
        
        # Calculate how rectangular the contour is
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        rect_ratio = area / box_area if box_area > 0 else 0
        
        # Draw large contours in blue
        cv2.drawContours(viz_img, [c], 0, (255, 0, 0), 2)
            
        # Try multiple epsilon values for polygon approximation
        for factor in [epsilon_factor, 0.01, 0.03, 0.05]:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, factor * perimeter, True)
            
            vertices = len(approx)
            
            # Calculate aspect ratio of the contour - prefer more rectangular shapes
            rect = cv2.boundingRect(approx)
            aspect_ratio = max(rect[2], rect[3]) / min(rect[2], rect[3]) if min(rect[2], rect[3]) > 0 else 1
            
            # Score the contour based on area, rectangularity and aspect ratio
            # We want large, rectangular contours with reasonable aspect ratios
            contour_score = area * rect_ratio * (1.0 / max(1.0, aspect_ratio))
            
            # Ideal case: 4-sided contour with good properties
            if vertices == 4:
                # Draw the approximated contour in green
                cv2.drawContours(viz_img, [approx], 0, (0, 255, 0), 2)
                
                # Draw the corners as circles
                for point in approx:
                    x, y = point[0]
                    cv2.circle(viz_img, (x, y), 5, (0, 255, 255), -1)
                
                possible_docs.append((approx, contour_score))
                # Add label
                cx = np.mean(approx[:, 0, 0])
                cy = np.mean(approx[:, 0, 1])
                cv2.putText(viz_img, f"Doc (score: {contour_score:.1f})", (int(cx)-50, int(cy)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break
                
            # Handle 3, 5, or 6 vertices cases
            elif vertices >= 3 and vertices <= 6:
                # Draw the original approximation in yellow
                cv2.drawContours(viz_img, [approx], 0, (0, 255, 255), 1)
                
                # Convert to a rectangle
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Draw the rectangular approximation in green
                cv2.drawContours(viz_img, [box], 0, (0, 255, 0), 2)
                
                # Draw the corners as circles
                for point in box:
                    x, y = point
                    cv2.circle(viz_img, (int(x), int(y)), 5, (0, 255, 255), -1)
                
                possible_docs.append((box.reshape(-1, 1, 2), contour_score))
                # Add label
                cx = np.mean(box[:, 0])
                cy = np.mean(box[:, 1])
                cv2.putText(viz_img, f"Doc (v={vertices}, score={contour_score:.1f})", 
                            (int(cx)-80, int(cy)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break
    
    # If we found possible documents, select the best one based on score
    if possible_docs:
        # Sort by score (highest first)
        possible_docs.sort(key=lambda x: x[1], reverse=True)
        best_docs = [doc[0] for doc in possible_docs]
        
        # Draw the best contour with a thick line
        cv2.drawContours(viz_img, [best_docs[0]], 0, (0, 255, 0), 3)
        cv2.putText(viz_img, "BEST MATCH", 
                    (int(viz_img.shape[1]/2)-50, int(viz_img.shape[0]/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return best_docs, viz_img
    
    return None, viz_img

def enhance_for_threshold(img):
    """Apply special preprocessing for threshold method to better capture full document edges"""
    # Create a border around the image to help with documents that extend to the edges
    bordered = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    
    # Increase contrast to make the document stand out
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(bordered)
    
    # Apply bilateral filter to preserve edges while smoothing
    bilateral = cv2.bilateralFilter(contrast, 9, 75, 75)
    
    return bilateral

def four_point_warp(img, pts):
    """Applies perspective warp to an image based on 4 points."""
    # Ensure pts are properly ordered
    rect = order_pts(pts)
    (tl, tr, br, bl) = rect

    # Draw the ordered points on a copy for visualization
    viz_img = img.copy()
    points = [tl, tr, br, bl]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
    labels = ["TL", "TR", "BR", "BL"]
    
    for (point, color, label) in zip(points, colors, labels):
        x, y = point.astype(int)
        cv2.circle(viz_img, (x, y), 10, color, -1)
        cv2.putText(viz_img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Compute width: max distance between br/bl or tr/tl
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(1, int(max(widthA, widthB)))

    # Compute height: max distance between tr/br or tl/bl
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(1, int(max(heightA, heightB)))

    # Destination points for "bird's eye view"
    dst = np.array([
        [0, 0],                  # Top-left
        [maxWidth - 1, 0],       # Top-right
        [maxWidth - 1, maxHeight - 1], # Bottom-right
        [0, maxHeight - 1]       # Bottom-left
    ], dtype="float32")

    # Draw the destination rectangle on viz_img
    dst_scaled = dst.copy()
    # Scale and shift the destination to avoid overlapping with the original
    scale_factor = 0.3
    shift_x = img.shape[1] - int(maxWidth * scale_factor) - 10
    shift_y = 10
    
    dst_scaled = dst_scaled * scale_factor
    dst_scaled[:, 0] += shift_x
    dst_scaled[:, 1] += shift_y
    dst_scaled = dst_scaled.astype(int)
    
    cv2.polylines(viz_img, [dst_scaled], True, (0, 255, 0), 2)
    cv2.putText(viz_img, "Target", (shift_x, shift_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Compute the perspective transform matrix and apply it
    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped, viz_img
    except cv2.error as e:
        print(f"Error during perspective warp: {e}")
        return img, viz_img

# ────────────────────────── Main scanning function ──────────────────────────

def scan(path, visualize=True, out="scanned_output.jpeg"):
    """
    Document scanning function with detailed visualization of the detection process.
    
    Args:
        path: Path to the input image
        visualize: Whether to show visualization windows
        out: Output file path
    """
    # Load the image
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Cannot open or read image file {path}")
        orig = img.copy()
    except Exception as e:
        sys.exit(f"Error loading image: {e}")
        
    # Display the original image
    if visualize:
        cv2.imshow("Original Image", img)
        cv2.waitKey(1000)  # Wait 1 second to ensure window is visible

    # Resize for faster processing
    proc_h = 700
    if img.shape[0] <= proc_h:
        ratio = 1.0
        small = img.copy()
    else:
        ratio = img.shape[0] / proc_h
        small = cv2.resize(img, (int(img.shape[1] / ratio), proc_h), 
                          interpolation=cv2.INTER_AREA)
                          
    # Display the resized image
    if visualize:
        cv2.imshow("Resized Image", small)
        cv2.waitKey(1000)

    # Convert to grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    if visualize:
        cv2.imshow("Grayscale", gray)
        cv2.waitKey(1000)

    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    if visualize:
        cv2.imshow("Enhanced Contrast", enhanced)
        cv2.waitKey(1000)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    if visualize:
        cv2.imshow("Blurred", blurred)
        cv2.waitKey(1000)

    # Store results from both methods
    edge_document_contour = None
    threshold_document_contour = None
    
    # ---------------------- EDGE-BASED DETECTION ----------------------
    print("\n====== METHOD 1: EDGE DETECTION ======")
    
    # Create a window for edge detection steps
    if visualize:
        cv2.namedWindow("Edge Detection Steps", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Edge Detection Steps", 800, 600)
    
    # Try different parameters for edge detection
    edge_params = [
        {"sigma": 0.33, "kernel_size": 5, "area_ratio": 0.1},
        {"sigma": 0.25, "kernel_size": 7, "area_ratio": 0.1},
        {"sigma": 0.40, "kernel_size": 5, "area_ratio": 0.05}
    ]
    
    for params in edge_params:
        print(f"\nTrying edge detection with: sigma={params['sigma']}, kernel={params['kernel_size']}, area_ratio={params['area_ratio']}")
        
        # Apply Canny edge detection
        edges = auto_canny(blurred, params['sigma'])
        
        if visualize:
            cv2.imshow("Canny Edges", edges)
            cv2.waitKey(1000)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params['kernel_size'], params['kernel_size']))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        if visualize:
            cv2.imshow("Morphological Closing", closed)
            cv2.waitKey(1000)
        
        # Apply dilation to connect broken edges
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        if visualize:
            cv2.imshow("Dilated Edges", dilated)
            cv2.waitKey(1000)
        
        # Find and visualize document contours
        contours, viz = find_document_contours(dilated, params['area_ratio'])
        
        if visualize:
            cv2.imshow("Edge Detection Steps", viz)
            cv2.waitKey(1000)
        
        if contours:
            edge_document_contour = contours[0]
            print(f"✓ Found document contour using edge detection!")
            break
    
    # ---------------------- THRESHOLD-BASED DETECTION ----------------------
    print("\n====== METHOD 2: THRESHOLD DETECTION ======")
    
    if visualize:
        cv2.namedWindow("Threshold Detection Steps", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Threshold Detection Steps", 800, 600)
    
    # Enhanced preprocessing for threshold method
    print("Applying additional preprocessing for threshold method...")
    
    # Apply special enhancement for threshold method to capture full document boundaries
    enhanced_for_threshold = enhance_for_threshold(gray)
    
    if visualize:
        cv2.imshow("Enhanced for Threshold", enhanced_for_threshold)
        cv2.waitKey(800)
    
    # Create multiple preprocessing versions for threshold
    preproc_versions = []
    
    # 1. Enhanced version specifically for threshold
    preproc_versions.append(("enhanced_threshold", enhanced_for_threshold))
    
    # 2. Standard blurred
    preproc_versions.append(("standard", blurred))
    
    # 3. Extra contrast enhancement
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    high_contrast = clahe2.apply(gray)
    preproc_versions.append(("high_contrast", high_contrast))
    
    # 4. Normalization
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    preproc_versions.append(("normalized", normalized))
    
    # 5. Sharpened
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    preproc_versions.append(("sharpened", sharpened))
    
    # Try full-document detection first with special parameters
    full_doc_params = [
        {"method": "otsu", "area_ratio": 0.2, "preprocessing": "enhanced_threshold"},
        {"method": "adaptive", "block_size": 31, "C": 11, "area_ratio": 0.15, "preprocessing": "enhanced_threshold"},
    ]
    
    # Try to detect full document first
    print("\nAttempting to detect full document...")
    for params in full_doc_params:
        preproc_img = next((img for name, img in preproc_versions if name == params["preprocessing"]), enhanced_for_threshold)
        
        if params["method"] == "adaptive":
            print(f"  Trying adaptive threshold: block_size={params['block_size']}, C={params['C']}, area_ratio={params['area_ratio']}")
            thresh = cv2.adaptiveThreshold(preproc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, params['block_size'], params['C'])
        elif params["method"] == "otsu":
            print(f"  Trying Otsu threshold with area_ratio={params['area_ratio']}")
            _, thresh = cv2.threshold(preproc_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        if visualize:
            cv2.imshow("Full Document Threshold", thresh)
            cv2.waitKey(800)
            
        # Close small holes to create a more solid document shape
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        if visualize:
            cv2.imshow("Closed for Full Document", closed)
            cv2.waitKey(800)
            
        # Find document contours with high area ratio
        contours, viz = find_document_contours(closed, params['area_ratio'])
        
        if visualize:
            cv2.imshow("Full Document Detection", viz)
            cv2.waitKey(1000)
            
        if contours:
            threshold_document_contour = contours[0]
            print(f"✓ Found FULL document contour using threshold method!")
            break
    
    # If full document detection failed, try the detailed approach
    if threshold_document_contour is None:
        # Try different parameters for thresholding with more combinations
        threshold_params = [
            {"method": "adaptive", "block_size": 11, "C": 5, "area_ratio": 0.05},
            {"method": "adaptive", "block_size": 21, "C": 10, "area_ratio": 0.05},
            {"method": "adaptive", "block_size": 31, "C": 15, "area_ratio": 0.05},
            {"method": "adaptive", "block_size": 15, "C": 3, "area_ratio": 0.03},
            {"method": "adaptive", "block_size": 25, "C": 8, "area_ratio": 0.03},
            {"method": "otsu", "area_ratio": 0.03},
            {"method": "simple", "value": 127, "area_ratio": 0.03},
            {"method": "simple", "value": 150, "area_ratio": 0.03},
            {"method": "simple", "value": 100, "area_ratio": 0.03}
        ]
        
        for preproc_name, preproc_img in preproc_versions:
            if threshold_document_contour is not None:
                break
            
            print(f"\nTrying threshold detection with {preproc_name} preprocessing...")
            
            if visualize:
                cv2.imshow("Preprocessing", preproc_img)
                cv2.waitKey(800)
            
            for params in threshold_params:
                if threshold_document_contour is not None:
                    break
                
                if params["method"] == "adaptive":
                    print(f"  Trying adaptive threshold: block_size={params['block_size']}, C={params['C']}, area_ratio={params['area_ratio']}")
                    thresh = cv2.adaptiveThreshold(preproc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, params['block_size'], params['C'])
                elif params["method"] == "otsu":
                    print(f"  Trying Otsu threshold with area_ratio={params['area_ratio']}")
                    _, thresh = cv2.threshold(preproc_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:  # simple
                    print(f"  Trying simple threshold with value={params['value']}, area_ratio={params['area_ratio']}")
                    _, thresh = cv2.threshold(preproc_img, params['value'], 255, cv2.THRESH_BINARY_INV)
                
                if visualize:
                    cv2.imshow("Threshold Result", thresh)
                    cv2.waitKey(800)
                  
                # Try multiple morphological operations to clean up threshold result
                kernel_sizes = [(3, 3), (5, 5)]
                
                for k_size in kernel_sizes:
                    if threshold_document_contour is not None:
                        break
                        
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
                    
                    # Try different combinations of morphological operations
                    morph_operations = [
                        ("open+close", lambda img: cv2.morphologyEx(
                            cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1), 
                            cv2.MORPH_CLOSE, kernel, iterations=2)),
                        ("close+open", lambda img: cv2.morphologyEx(
                            cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2), 
                            cv2.MORPH_OPEN, kernel, iterations=1)),
                        ("dilate", lambda img: cv2.dilate(img, kernel, iterations=2)),
                        ("erode+dilate", lambda img: cv2.dilate(
                            cv2.erode(img, kernel, iterations=1), kernel, iterations=3))
                    ]
                    
                    for op_name, operation in morph_operations:
                        if threshold_document_contour is not None:
                            break
                            
                        print(f"    Trying {op_name} morphology with kernel {k_size}")
                        cleaned = operation(thresh)
                        
                        if visualize:
                            cv2.imshow(f"Cleaned ({op_name})", cleaned)
                            cv2.waitKey(800)
                        
                        # Find and visualize document contours with relaxed parameters
                        for epsilon in [0.02, 0.03, 0.05, 0.08]:
                            for area_ratio in [params['area_ratio'], params['area_ratio']*0.5, 0.02]:
                                # Try with both normal and inverted images
                                for img_version, img_name in [(cleaned, "normal"), (cv2.bitwise_not(cleaned), "inverted")]:
                                    if threshold_document_contour is not None:
                                        break
                                        
                                    contours, viz = find_document_contours(img_version, area_ratio, epsilon)
                                    
                                    if visualize:
                                        cv2.imshow(f"Threshold Detection Steps ({img_name})", viz)
                                        cv2.waitKey(800)
                                    
                                    if contours:
                                        threshold_document_contour = contours[0]
                                        print(f"✓ Found document contour using threshold method!")
                                        print(f"  Method: {params['method']}, Preprocessing: {preproc_name}")
                                        print(f"  Morphology: {op_name}, epsilon={epsilon}, area_ratio={area_ratio}")
                                        break
    
    # Process results from both methods
    results = []
    
    # ---------------------- PROCESS BOTH DETECTION RESULTS ----------------------
    print("\n====== PROCESSING RESULTS FROM BOTH METHODS ======")
    
    # Process edge detection result if available
    if edge_document_contour is not None:
        print("\nProcessing edge detection result...")
        # Reshape and scale the points back to original image
        edge_points = edge_document_contour.reshape(4, 2).astype("float32") * ratio
        
        # Warp the document
        edge_warped, edge_warp_viz = four_point_warp(orig, edge_points)
        
        if visualize:
            cv2.namedWindow("Edge Detection - Document Corners", cv2.WINDOW_NORMAL)
            cv2.imshow("Edge Detection - Document Corners", edge_warp_viz)
            cv2.waitKey(1000)
            
            cv2.namedWindow("Edge Detection - Warped Document", cv2.WINDOW_NORMAL)
            cv2.imshow("Edge Detection - Warped Document", edge_warped)
            cv2.waitKey(1000)
        
        # Convert to grayscale
        edge_warped_gray = cv2.cvtColor(edge_warped, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for scan effect
        edge_scan_result = cv2.adaptiveThreshold(edge_warped_gray, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 21, 10)
        
        # Save the edge detection result
        edge_out_path = out.replace(".jpeg", "_edge.jpeg").replace(".jpg", "_edge.jpg")
        try:
            cv2.imwrite(edge_out_path, edge_scan_result)
            print(f"✅ Edge detection scan saved as {edge_out_path}")
            results.append(("Edge", edge_scan_result, edge_out_path))
        except Exception as e:
            print(f"Error saving edge detection output: {e}")
    else:
        print("\n❌ Edge detection method did not find a document contour.")
    
    # Process threshold detection result if available
    if threshold_document_contour is not None:
        print("\nProcessing threshold detection result...")
        # Reshape and scale the points back to original image
        threshold_points = threshold_document_contour.reshape(4, 2).astype("float32") * ratio
        
        # Warp the document
        threshold_warped, threshold_warp_viz = four_point_warp(orig, threshold_points)
        
        if visualize:
            cv2.namedWindow("Threshold Detection - Document Corners", cv2.WINDOW_NORMAL)
            cv2.imshow("Threshold Detection - Document Corners", threshold_warp_viz)
            cv2.waitKey(1000)
            
            cv2.namedWindow("Threshold Detection - Warped Document", cv2.WINDOW_NORMAL)
            cv2.imshow("Threshold Detection - Warped Document", threshold_warped)
            cv2.waitKey(1000)
        
        # Convert to grayscale
        threshold_warped_gray = cv2.cvtColor(threshold_warped, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for scan effect
        threshold_scan_result = cv2.adaptiveThreshold(threshold_warped_gray, 255, 
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 21, 10)
        
        # Save the threshold detection result
        threshold_out_path = out.replace(".jpeg", "_threshold.jpeg").replace(".jpg", "_threshold.jpg")
        try:
            cv2.imwrite(threshold_out_path, threshold_scan_result)
            print(f"✅ Threshold detection scan saved as {threshold_out_path}")
            results.append(("Threshold", threshold_scan_result, threshold_out_path))
        except Exception as e:
            print(f"Error saving threshold detection output: {e}")
    else:
        print("\n❌ Threshold detection method did not find a document contour.")
    
    # Check if any detection method succeeded
    if not results:
        if visualize:
            cv2.destroyAllWindows()
        sys.exit("\n❌ Both detection methods failed. Try a different image with clearer document edges.")
    
    # Display results side by side for comparison
    if visualize and len(results) > 1:
        print("\nDisplaying both results side by side for comparison. Press any key to exit...")
        
        # Create a combined view of both results
        edge_result = next((result for result in results if result[0] == "Edge"), None)
        threshold_result = next((result for result in results if result[0] == "Threshold"), None)
        
        if edge_result and threshold_result:
            # Resize both to the same height if needed
            h1, w1 = edge_result[1].shape[:2]
            h2, w2 = threshold_result[1].shape[:2]
            
            target_h = min(h1, h2, 800)  # Cap at 800px height for display
            
            # Scale while maintaining aspect ratio
            edge_display = cv2.resize(edge_result[1], 
                                    (int(w1 * target_h / h1), target_h))
            threshold_display = cv2.resize(threshold_result[1], 
                                         (int(w2 * target_h / h2), target_h))
            
            # Convert to BGR for concatenation if grayscale
            if len(edge_display.shape) == 2:
                edge_display = cv2.cvtColor(edge_display, cv2.COLOR_GRAY2BGR)
            if len(threshold_display.shape) == 2:
                threshold_display = cv2.cvtColor(threshold_display, cv2.COLOR_GRAY2BGR)
            
            # Create a 50px blank space between images with labels
            spacer = np.ones((target_h, 50, 3), dtype=np.uint8) * 255
            
            # Add titles to the images
            cv2.putText(edge_display, "Edge Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(threshold_display, "Threshold Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Combine images with spacer
            comparison = np.hstack((edge_display, spacer, threshold_display))
            
            cv2.namedWindow("Comparison of Methods", cv2.WINDOW_NORMAL)
            cv2.imshow("Comparison of Methods", comparison)
            cv2.waitKey(0)
    elif visualize and len(results) == 1:
        # If only one detection method worked, show that result
        method, result, _ = results[0]
        cv2.imshow(f"Final Scan ({method} Detection)", result)
        print(f"\nShowing {method} detection result. Press any key to exit...")
        cv2.waitKey(0)
    
    # Clean up
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensure all windows close properly

# ────────────────────────── Command line interface ──────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document Scanner with visualization of edge and threshold detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("-o", "--output", default="scanned_output.jpeg",
                       help="Path for output scanned image")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization windows")
    
    args = parser.parse_args()
    
    input_path = Path(args.image)
    if not input_path.is_file():
        sys.exit(f"Error: Input image file not found at {args.image}")
    
    scan(input_path, visualize=not args.no_viz, out=args.output) 