#!/usr/bin/env python3
"""
Robust automatic document scanner with edge and threshold-based detection.
Output file: sample_output.jpeg
"""

import cv2
import sys
import argparse
import textwrap
import itertools
import numpy as np
from pathlib import Path

# ────────────────────────── tiny helpers ──────────────────────────

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
            # Don't return early - create a default rectangle instead
            h, w = 100, 100  # Default size
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

def four_point_warp(img, pts):
    """Applies perspective warp to an image based on 4 points."""
    # Ensure pts are properly ordered
    rect = order_pts(pts)
    (tl, tr, br, bl) = rect

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

    # Compute the perspective transform matrix and apply it
    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped
    except cv2.error as e:
        print(f"Error during perspective warp: {e}")
        return img

def find_page_contours(img_processed, min_area_ratio=0.1):
    """
    Finds contours in a processed (binary/edge) image, filters them,
    and returns potential page contours (approximated to 4 points).
    """
    cnts, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    img_area = img_processed.shape[0] * img_processed.shape[1]
    min_area = img_area * min_area_ratio

    possible_pages = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            break

        # Try multiple epsilon values for polygon approximation
        for epsilon_factor in [0.02, 0.01, 0.03, 0.05]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon_factor * peri, True)
            
            # Accept contours with 3-6 vertices and fix to 4 sides
            vertices = len(approx)
            
            # Ideal case: exactly 4 vertices
            if vertices == 4:
                possible_pages.append(approx)
                return possible_pages  # Found ideal contour
            # Also accept 3-6 sided shapes and convert to 4 sides
            elif vertices >= 3 and vertices <= 6:
                # Convert to minimum area rectangle
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                possible_pages.append(box.reshape(-1, 1, 2))
                return possible_pages

    return possible_pages

# ────────────────────────── main routine ──────────────────────────
def scan(path, debug=False, out="sample_output.jpeg"):
    """Main fully automatic scanning function."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Cannot open or read image file {path}")
        orig = img.copy()
    except Exception as e:
        sys.exit(f"Error loading image: {e}")

    # --- Preprocessing for Detection ---
    proc_h = 700
    if img.shape[0] <= proc_h:
        ratio = 1.0
        small = img.copy()
    else:
        ratio = img.shape[0] / proc_h
        try:
            small = cv2.resize(img, (int(img.shape[1] / ratio), proc_h), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            sys.exit(f"Error resizing image: {e}")

    # Create more preprocessing variations
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g)
    
    # Create multiple blur versions for more robust detection
    blur_versions = [
        ("standard", cv2.GaussianBlur(g_enhanced, (5, 5), 0)),
        ("light", cv2.GaussianBlur(g_enhanced, (3, 3), 0)),
        ("heavy", cv2.GaussianBlur(g_enhanced, (7, 7), 0)),
        ("median", cv2.medianBlur(g_enhanced, 5))  # Better for noise
    ]

    page_contour = None

    # --- Attempt 1: Edge-Based Detection ---
    print("Attempting automatic detection (Method 1: Edge-based)...")
    sigmas = [0.33, 0.25, 0.4, 0.5, 0.6]  # More varied sigmas
    kernels = [5, 7, 9]
    min_area_ratios = [0.2, 0.1, 0.05, 0.02]  # Added smaller area ratio for documents

    found_edge_based = False
    
    # Try each blur version
    for blur_name, blur in blur_versions:
        if found_edge_based:
            break
            
        for sigma, kernel, min_area_ratio in itertools.product(sigmas, kernels, min_area_ratios):
            # Generate edges
            edges = auto_canny(blur, sigma)
            
            # Try different morphological operations
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
            
            # 1. Standard close then dilate
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
            dilated = cv2.dilate(closed, k, iterations=1)
            
            # 2. Heavy dilation for connecting broken edges
            dilated2 = cv2.dilate(edges, k, iterations=2)
            
            # Try each version
            for version_name, processed in [("dilated", dilated), ("heavy_dilated", dilated2), ("closed", closed)]:
                candidates = find_page_contours(processed, min_area_ratio)

                if debug:
                    dbg = small.copy()
                    cv2.imshow("Edges", edges)
                    cv2.imshow("Processed", processed)
                    if candidates:
                        cv2.drawContours(dbg, candidates, -1, (0, 255, 0), 2)
                    cv2.imshow(f"Edge: {blur_name}, s={sigma}, k={kernel}, r={min_area_ratio}, version={version_name}", dbg)
                    if cv2.waitKey(0) == 27: pass
                    cv2.destroyWindow("Edges")
                    cv2.destroyWindow("Processed")
                    cv2.destroyWindow(f"Edge: {blur_name}, s={sigma}, k={kernel}, r={min_area_ratio}")

                if candidates:
                    page_contour = candidates[0]
                    print(f"  ✓ Found contour (Edge-based) with {blur_name} blur, sigma={sigma}, kernel={kernel}, ratio={min_area_ratio}, version={version_name}")
                    found_edge_based = True
                    break
            
            if found_edge_based:
                break

    if debug: cv2.destroyAllWindows()

    # --- Attempt 2: Threshold-Based Detection (if edge-based failed) ---
    if not found_edge_based:
        print("Attempting automatic detection (Method 2: Threshold-based)...")
        
        # Try each blur version
        for blur_name, blur in blur_versions:
            if page_contour is not None:
                break
                
            # Try different thresholding approaches
            thresholds = []
            
            # 1. Adaptive threshold with different parameters
            for block_size in [11, 21, 31]:
                for c_val in [5, 10, 15]:
                    # Try both normal and inverted
                    thresh1 = cv2.adaptiveThreshold(blur, 255,
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, block_size, c_val)
                    thresholds.append(("AdaptiveINV", thresh1, block_size, c_val))
                    
                    thresh2 = cv2.adaptiveThreshold(blur, 255,
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, block_size, c_val)
                    thresholds.append(("Adaptive", thresh2, block_size, c_val))
            
            # 2. Otsu thresholding (global optimal threshold)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresholds.append(("Otsu", otsu, 0, 0))
            
            # 3. Simple global threshold at different levels
            for t in [127, 150, 180]:
                _, simple = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
                thresholds.append(("Simple", simple, 0, t))
            
            for thresh_type, thresh, block_size, c_val in thresholds:
                # Try different morphology operations
                k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                k_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                
                # Different cleaning approaches
                cleaned_versions = [
                    ("basic", cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_small, iterations=1)),
                    ("thorough", cv2.morphologyEx(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_small), 
                                                cv2.MORPH_CLOSE, k_medium))
                ]
                
                for clean_type, cleaned in cleaned_versions:
                    candidates = find_page_contours(cleaned, min_area_ratio=0.05)
                    
                    if debug:
                        dbg = small.copy()
                        cv2.imshow("Threshold", thresh)
                        cv2.imshow("Cleaned", cleaned)
                        if candidates:
                            cv2.drawContours(dbg, candidates, -1, (0, 0, 255), 2)
                        title = f"Thresh: {blur_name}, {thresh_type}, clean={clean_type}"
                        cv2.imshow(title, dbg)
                        if cv2.waitKey(0) == 27: pass
                        cv2.destroyWindow("Threshold")
                        cv2.destroyWindow("Cleaned")
                        cv2.destroyWindow(title)
                    
                    if candidates:
                        page_contour = candidates[0]
                        print(f"  ✓ Found contour (Threshold: {thresh_type}, {clean_type}) with {blur_name} blur")
                        if thresh_type in ["AdaptiveINV", "Adaptive"]:
                            print(f"    Using block_size={block_size}, C={c_val}")
                        elif thresh_type == "Simple":
                            print(f"    Using threshold={c_val}")
                        break
                
                if page_contour is not None:
                    break

    if debug: 
        cv2.destroyAllWindows()
        
    # --- Attempt 3: Last resort - heuristic detection ---
    if page_contour is None:
        print("Attempting automatic detection (Method 3: Heuristic-based)...")
        
        # Use a more aggressive approach - convert to binary and find the largest non-white rectangle
        for blur_name, blur in blur_versions:
            # Convert to binary image with reasonable threshold
            _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours of all non-white regions
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour by area
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                # Only consider if it's reasonably big
                img_area = binary.shape[0] * binary.shape[1]
                if area > img_area * 0.1:
                    # Get minimum area rectangle
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    page_contour = np.int32(box).reshape(4, 1, 2)
                    print(f"  ✓ Found contour (Heuristic) with {blur_name} blur")
                    break

    # --- If all detection methods fail after trying everything ---
    if page_contour is None:
        print("\n⚠️  Detection failed despite trying all parameters and preprocessing options.")
        print("    Consider the following tips:")
        print("    - Ensure your image has good lighting and contrast")
        print("    - Remove reflections or shadows if possible")
        print("    - Try cropping the image closer to the document")
        print("    - For best results, capture the image with the document against a contrasting background")
        sys.exit("Exiting without processing. No document contours found.")

    # Get the points from the detected contour
    page_points = page_contour.reshape(4, 2).astype("float32")

    # --- Warping and Post-processing ---
    final_points = page_points * ratio
    warped = four_point_warp(orig, final_points)

    # Convert warped image to grayscale for thresholding
    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for final scan effect
    scan_final = cv2.adaptiveThreshold(wgray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 21, 10)

    # --- Output ---
    out_path = Path(out)
    try:
        if scan_final is None or scan_final.size == 0:
             raise ValueError("Final scanned image is empty or invalid.")
        cv2.imwrite(str(out_path), scan_final)
        print(f"✅ Scan saved successfully as {out_path.name}")
    except (cv2.error, ValueError, IOError) as e:
        sys.exit(f"Error: Could not save the image to {out}. Reason: {e}")

    # Display the final scanned image
    try:
        cv2.imshow("Scanned Document", scan_final)
        print("Displaying final scan. Press any key to exit.")
        cv2.waitKey(0)
    except cv2.error as e:
         print(f"Could not display the final image: {e}")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

# ────────────────────────── CLI ──────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fully automatic document scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python testing.py sample.jpeg
          python testing.py my_photo.png --debug
          python testing.py document.jpg --output processed_doc.png
        '''))
    ap.add_argument("image", help="Path to the input photo (e.g., sample.jpeg)")
    ap.add_argument("-o", "--output", default="sample_output.jpeg",
                    help="Path for the output scanned image (default: sample_output.jpeg)")
    ap.add_argument("--debug", action="store_true",
                    help="Show intermediate steps of the detection process")

    ns = ap.parse_args()

    input_path = Path(ns.image)
    if not input_path.is_file():
        sys.exit(f"Error: Input image file not found at {ns.image}")
    scan(input_path, debug=ns.debug, out=ns.output)
