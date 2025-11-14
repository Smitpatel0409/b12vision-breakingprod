"""
Vitamin B12 Hand Image Analysis Module

This module processes hand images to determine Vitamin B12 status by:
1. Detecting and orienting the hand correctly using MediaPipe
2. Extracting specific regions of interest (ROIs) on the index finger
3. Analyzing color differences between nail bed and finger joint
4. Comparing colors against a reference shade database
5. Determining B12 status based on color score differences

The analysis is based on the principle that B12 deficiency can cause
color changes in nail beds and surrounding tissue.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
import mediapipe as mp
import os
from typing import Optional, Tuple

# Global variable to store reference shade data
# Loaded from hsv_values.txt file
shade_data = []


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a specified angle without cropping.
    
    This function calculates the new bounding box size needed to fit the
    entire rotated image, preventing any part from being cut off.
    
    Args:
        image (np.ndarray): Input image to rotate
        angle (float): Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        np.ndarray: Rotated image with adjusted dimensions
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Calculate new dimensions to accommodate rotated image
    angle_rad = np.abs(np.radians(angle))
    new_width = int(abs(width * np.cos(angle_rad)) + abs(height * np.sin(angle_rad)))
    new_height = int(abs(width * np.sin(angle_rad)) + abs(height * np.cos(angle_rad)))

    # Create rotation matrix centered on image center
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Adjust translation to center the rotated image in new bounding box
    rotation_matrix[0, 2] += (new_width - width) // 2
    rotation_matrix[1, 2] += (new_height - height) // 2

    # Apply rotation with new dimensions
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    return rotated_image


def load_hsv_values(file_path="hsv_values.txt"):
    """
    Load reference HSV color values from a text file.
    
    The file should contain HSV values in the format:
    (360°, 100%, 100%)
    (180°, 50%, 75%)
    etc.
    
    These values represent a reference scale for comparing hand colors.
    Each line represents a shade level, with line number corresponding to
    the shade number in the analysis.
    
    Args:
        file_path (str): Path to HSV values file
        
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    global shade_data
    shade_data = []
    
    # Ensure file path is relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_file_path = os.path.join(script_dir, file_path)

    # Check if file exists
    if not os.path.exists(absolute_file_path):
        return False, f"HSV file not found: {absolute_file_path}"
    
    try:
        # Read and parse HSV values
        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return False, "HSV file is empty."
            
            # Process each line
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Parse HSV values from format: (H, S%, V%)
                hsv_str = line.strip('()').split(',')
                if len(hsv_str) != 3:
                    continue
                
                try:
                    # Convert HSV from human-readable format to OpenCV format
                    # H: 0-360° → 0-179, S: 0-100% → 0-255, V: 0-100% → 0-255
                    hue = int(float(hsv_str[0].replace('°', '').replace('Â', '').strip()) * 179 // 360)
                    sat = int(float(hsv_str[1].replace('%', '').strip()) * 255 // 100)
                    val = int(float(hsv_str[2].replace('%', '').strip()) * 255 // 100)
                    
                    dominant_color = np.array([hue, sat, val], dtype=np.uint8)
                    shade_data.append({
                        "number": line_num,
                        "dominant_color": dominant_color
                    })
                except (ValueError, TypeError):
                    # Skip lines that can't be parsed
                    continue
        
        # Ensure we loaded at least some valid data
        if not shade_data:
            return False, "No valid HSV values found in file."
        
        print(f"✓ Loaded {len(shade_data)} reference shades from {file_path}")
        return True, None
    
    except UnicodeDecodeError:
        return False, "Invalid file encoding. Ensure file is UTF-8 encoded."
    except PermissionError:
        return False, f"Permission denied accessing file: {absolute_file_path}"
    except Exception as e:
        return False, f"Error reading HSV file: {str(e)}"


def compare_dominant_color(hand_roi_hsv: np.ndarray) -> Tuple[Optional[dict], float]:
    """
    Find the best matching shade for a given hand region.
    
    Uses K-means clustering to find the dominant color in the ROI,
    then compares it against all reference shades to find the closest match.
    
    Args:
        hand_roi_hsv (np.ndarray): ROI from hand image in HSV color space
        
    Returns:
        tuple: (best_match: dict or None, distance: float)
               best_match contains 'number' and 'dominant_color'
               distance is the Euclidean distance in HSV space
    """
    # Check if ROI is empty
    if hand_roi_hsv.size == 0:
        return None, float('inf')
    
    # Reshape ROI for K-means clustering
    roi_reshaped = hand_roi_hsv.reshape(-1, 3)
    
    # Use K-means to find dominant color (k=1 cluster)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, centers = cv2.kmeans(
        roi_reshaped.astype(np.float32),
        1,  # Find 1 dominant color
        None,
        criteria,
        10,  # 10 attempts
        cv2.KMEANS_RANDOM_CENTERS
    )
    dominant_hand = centers[0].astype(int)
    
    # Find closest matching shade from reference data
    best_match, min_distance = None, float('inf')
    for shade in shade_data:
        # Calculate Euclidean distance in HSV space
        distance = np.sqrt(np.sum((dominant_hand - shade['dominant_color']) ** 2))
        if distance < min_distance:
            min_distance = distance
            best_match = shade
    
    return best_match, min_distance


def check_lighting_quality(image_gray: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate image lighting conditions.
    
    Checks brightness and contrast to ensure image quality is sufficient
    for accurate color analysis.
    
    Args:
        image_gray (np.ndarray): Grayscale version of input image
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    # Calculate lighting metrics
    brightness = np.mean(image_gray)
    contrast = np.std(image_gray)
    
    # Check if image is too dark
    if brightness < 20:
        return False, "Image too dark. Use brighter, even lighting."
    
    # Check if image has insufficient contrast (flat/washed out)
    if contrast < 20:
        return False, "Image too flat. Use better lighting contrast."
    
    # Check if image has excessive contrast (harsh shadows)
    if contrast > 85:
        return False, "Image too harsh. Use softer, natural light."
    
    print(f"✓ Lighting OK: brightness={brightness:.1f}, contrast={contrast:.1f}")
    return True, None


def detect_hand_orientation(hand_landmarks, image_shape) -> Tuple[float, bool]:
    """
    Determine hand orientation and whether it's a left or right hand.
    
    Uses MediaPipe landmarks to calculate:
    1. Hand orientation angle (which way fingers are pointing)
    2. Handedness (left vs right hand)
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        image_shape: Shape tuple of the image (height, width, channels)
        
    Returns:
        tuple: (rotation_angle: float, is_left_hand: bool)
               rotation_angle is degrees needed to make hand upright
    """
    # MediaPipe landmark indices
    THUMB_TIP, PINKY_TIP, WRIST, MIDDLE_MCP = 4, 20, 0, 9
    
    # Extract key landmark coordinates
    thumb_x = hand_landmarks.landmark[THUMB_TIP].x * image_shape[1]
    thumb_y = hand_landmarks.landmark[THUMB_TIP].y * image_shape[0]
    pinky_x = hand_landmarks.landmark[PINKY_TIP].x * image_shape[1]
    pinky_y = hand_landmarks.landmark[PINKY_TIP].y * image_shape[0]
    wrist_x = hand_landmarks.landmark[WRIST].x * image_shape[1]
    wrist_y = hand_landmarks.landmark[WRIST].y * image_shape[0]
    middle_mcp_x = hand_landmarks.landmark[MIDDLE_MCP].x * image_shape[1]
    middle_mcp_y = hand_landmarks.landmark[MIDDLE_MCP].y * image_shape[0]
    
    # Calculate hand angle from wrist to middle finger
    angle = np.arctan2(middle_mcp_y - wrist_y, middle_mcp_x - wrist_x) * 180 / np.pi
    if angle < 0:
        angle += 360  # Normalize to 0-360 range
    
    # Round to nearest 90° to determine primary orientation
    rounded_angle = round(angle / 90) * 90
    if rounded_angle == 360:
        rounded_angle = 0
    
    # Disambiguate 180° case (could be upside down OR fingers pointing left)
    if rounded_angle == 180:
        # Use thumb position relative to wrist for disambiguation
        if thumb_y < wrist_y:  # Thumb above wrist
            rounded_angle = 90  # Fingers pointing left
    
    # Map orientation to required rotation
    # 270°: Upright (no rotation needed)
    # 180°: Upside down (rotate 180°)
    # 0°: Fingers right (rotate 90° CCW)
    # 90°: Fingers left (rotate 90° CW)
    rotation_map = {270: 0, 180: 180, 0: 90, 90: -90}
    rotation_angle = rotation_map.get(rounded_angle, 0)
    
    # Determine handedness using cross product
    vector_wrist_to_thumb = np.array([thumb_x - wrist_x, thumb_y - wrist_y])
    vector_wrist_to_pinky = np.array([pinky_x - wrist_x, pinky_y - wrist_y])
    cross_product = np.cross(vector_wrist_to_thumb, vector_wrist_to_pinky)
    
    # MediaPipe mirrors handedness, so we flip the interpretation
    # Negative cross product typically means left hand (in MediaPipe's mirrored view)
    is_left_hand = cross_product < 0
    
    print(f"⚙ Hand orientation: {rounded_angle}°, rotation needed: {rotation_angle}°")
    print(f"⚙ Detected hand: {'LEFT' if is_left_hand else 'RIGHT'}")
    
    return rotation_angle, is_left_hand


def extract_rois(hand_landmarks, image_shape) -> Tuple[dict, dict]:
    """
    Extract two Regions of Interest (ROIs) on the index finger.
    
    Area A: Between DIP (distal interphalangeal) and PIP joints - nail bed region
    Area B: At PIP (proximal interphalangeal) joint - finger joint region
    
    The color difference between these areas indicates B12 status.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        image_shape: Shape of the image (height, width, channels)
        
    Returns:
        tuple: (area_a: dict, area_b: dict)
               Each dict contains: x, y_start, y_end, width, height
    """
    # MediaPipe landmark indices for index finger
    PIP_INDEX, DIP_INDEX, TIP_INDEX, MCP_INDEX = 6, 7, 8, 5
    
    # ROI size parameters (in pixels from center)
    ROI_SIZE_A, ROI_SIZE_B = 3, 3
    
    # Extract landmark positions in pixel coordinates
    pip_x = int(hand_landmarks.landmark[PIP_INDEX].x * image_shape[1])
    pip_y = int(hand_landmarks.landmark[PIP_INDEX].y * image_shape[0])
    dip_x = int(hand_landmarks.landmark[DIP_INDEX].x * image_shape[1])
    dip_y = int(hand_landmarks.landmark[DIP_INDEX].y * image_shape[0])
    tip_x = int(hand_landmarks.landmark[TIP_INDEX].x * image_shape[1])
    tip_y = int(hand_landmarks.landmark[TIP_INDEX].y * image_shape[0])
    mcp_x = int(hand_landmarks.landmark[MCP_INDEX].x * image_shape[1])
    mcp_y = int(hand_landmarks.landmark[MCP_INDEX].y * image_shape[0])
    
    # Calculate finger length for proportional ROI sizing
    finger_length = np.sqrt((tip_x - mcp_x) ** 2 + (tip_y - mcp_y) ** 2)
    print(f"⚙ Finger length (tip to MCP): {finger_length:.1f}px")
    
    # Position Area B at PIP joint (typically 44% from tip to MCP)
    pip_expected_y = tip_y + 0.44 * (mcp_y - tip_y)
    area_b = {
        'x': pip_x,
        'y': int(pip_expected_y),
        'width': ROI_SIZE_B * 2,
        'height': ROI_SIZE_B * 2
    }
    
    # Position Area A between DIP and PIP (around 30% from tip)
    # Height is proportional to finger length (7% of finger length)
    area_a_center_y = tip_y + 0.30 * (mcp_y - tip_y)
    area_a_height = int(0.07 * finger_length)
    area_a = {
        'x': (pip_x + dip_x) // 2,
        'y_start': max(0, int(area_a_center_y - area_a_height // 2)),
        'y_end': max(0, int(area_a_center_y + area_a_height // 2)),
        'width': ROI_SIZE_A * 2,
        'height': area_a_height
    }
    
    print(f"⚙ Area A (nail bed): x={area_a['x']}, y={area_a['y_start']}-{area_a['y_end']}")
    print(f"⚙ Area B (joint): x={area_b['x']}, y={area_b['y']}")
    
    return area_a, area_b


def validate_roi_bounds(area_a, area_b, image_shape, roi_size_a=3, roi_size_b=3) -> Tuple[bool, Optional[str]]:
    """
    Validate that ROI coordinates are within image boundaries.
    
    Args:
        area_a (dict): Area A coordinates
        area_b (dict): Area B coordinates
        image_shape: Image dimensions
        roi_size_a (int): Half-width of Area A
        roi_size_b (int): Half-width of Area B
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    # Check Area A bounds
    if (area_a['y_start'] >= area_a['y_end'] or 
        area_a['x'] - roi_size_a < 0 or 
        area_a['x'] + roi_size_a >= image_shape[1] or
        area_a['y_end'] >= image_shape[0]):
        return False, "Invalid ROI bounds for Area A (nail bed region)."
    
    # Check Area B bounds
    if (area_b['y'] - roi_size_b < 0 or 
        area_b['x'] - roi_size_b < 0 or 
        area_b['x'] + roi_size_b >= image_shape[1] or
        area_b['y'] + roi_size_b >= image_shape[0]):
        return False, "Invalid ROI bounds for Area B (joint region)."
    
    return True, None


def visualize_results(image_rgb, area_a, area_b, match_a_num, match_b_num, 
                     processed_image_path, roi_size_a=3, roi_size_b=3):
    """
    Create visualization of the analysis with annotated ROIs.
    
    Overlays colored rectangles on Areas A and B, and saves the result.
    
    Args:
        image_rgb (np.ndarray): Original image in RGB
        area_a (dict): Area A coordinates
        area_b (dict): Area B coordinates
        match_a_num (int): Shade number matched for Area A
        match_b_num (int): Shade number matched for Area B
        processed_image_path (str): Where to save the visualization
        roi_size_a (int): Half-width of Area A
        roi_size_b (int): Half-width of Area B
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    
    # Draw Area A (nail bed) in blue
    plt.gca().add_patch(plt.Rectangle(
        (area_a['x'] - roi_size_a, area_a['y_start']),
        roi_size_a * 2,
        area_a['y_end'] - area_a['y_start'],
        edgecolor='white',
        facecolor='blue',
        alpha=0.5,
        lw=3,
        label=f"Area (A) Match: {match_a_num}"
    ))
    
    # Draw Area B (joint) in green
    plt.gca().add_patch(plt.Rectangle(
        (area_b['x'] - roi_size_b, area_b['y'] - roi_size_b),
        roi_size_b * 2,
        roi_size_b * 2,
        edgecolor='white',
        facecolor='green',
        alpha=0.5,
        lw=3,
        label=f"Area (B) Match: {match_b_num}"
    ))
    
    # Mark centers with white dots
    area_a_y_center = (area_a['y_start'] + area_a['y_end']) // 2
    plt.scatter([area_b['x'], area_a['x']], 
               [area_b['y'], area_a_y_center], 
               c='white', s=50, edgecolors='black')
    
    # Add text labels
    plt.text(area_a['x'] + roi_size_a + 37, area_a_y_center, 
            'Area (A)', color='blue', fontsize=12)
    plt.text(area_b['x'] + roi_size_b + 37, area_b['y'], 
            'Area (B)', color='green', fontsize=12)
    
    # Remove axes for cleaner look
    plt.axis('off')
    
    # Save figure
    plt.savefig(processed_image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Saved visualization: {processed_image_path}")


def process_image(image_path: str, processed_folder: str = "processed") -> Tuple[Optional[str], str, Optional[float]]:
    """
    Main function to process a hand image and determine Vitamin B12 status.
    
    Complete workflow:
    1. Load HSV reference values if not already loaded
    2. Read and validate input image
    3. Check lighting quality
    4. Detect hand using MediaPipe
    5. Determine hand orientation and correct it
    6. Verify it's a left hand
    7. Extract ROIs (Areas A and B)
    8. Compare ROI colors against reference shades
    9. Calculate color score difference
    10. Determine B12 status
    11. Create visualization
    
    Args:
        image_path (str): Path to input hand image
        processed_folder (str): Folder to save processed image
        
    Returns:
        tuple: (processed_image_path: str or None,
                vitamin_b12_status: str,
                color_score_diff: float or None)
                
        If processing fails, returns (None, error_message, None)
    """
    global shade_data
    
    # Load reference shade data if not already loaded
    if not shade_data:
        success, error = load_hsv_values()
        if not success:
            return None, error or "Failed to load HSV values.", None
    
    # Ensure output folder exists
    os.makedirs(processed_folder, exist_ok=True)
    
    try:
        # Load input image
        print(f"⚙ Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            return None, "Unable to load image. Check the file path.", None
        
        # Convert to different color spaces for analysis
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Validate lighting conditions
        is_valid, error = check_lighting_quality(image_gray)
        if not is_valid:
            return None, error, None
        
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            
            # Detect hand landmarks
            print("⚙ Detecting hand landmarks...")
            results = hands.process(image_rgb)
            if not results.multi_hand_landmarks:
                return None, "No hand detected. Please provide an image with a visible left hand.", None
            
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Determine hand orientation and correct it
            rotation_angle, is_left_hand = detect_hand_orientation(hand_landmarks, image.shape)
            
            # Rotate image if needed
            if rotation_angle != 0:
                print(f"⚙ Rotating image by {rotation_angle}°...")
                image = rotate_image(image, rotation_angle)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Redetect hand after rotation
                results = hands.process(image_rgb)
                if not results.multi_hand_landmarks:
                    return None, "No hand detected after rotation.", None
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Recalculate handedness after rotation
                _, is_left_hand = detect_hand_orientation(hand_landmarks, image.shape)
            
            # Verify it's a left hand (required for consistent analysis)
            if not is_left_hand:
                return None, "Right hand detected. Please use your LEFT hand for analysis.", None
            
            # Extract Regions of Interest
            area_a, area_b = extract_rois(hand_landmarks, image.shape)
            
            # Validate ROI bounds
            ROI_SIZE_A, ROI_SIZE_B = 3, 3
            is_valid, error = validate_roi_bounds(area_a, area_b, image.shape, ROI_SIZE_A, ROI_SIZE_B)
            if not is_valid:
                return None, error, None
            
            # Extract ROI images in HSV color space
            area_a_roi_hsv = image_hsv[
                area_a['y_start']:area_a['y_end'],
                area_a['x'] - ROI_SIZE_A:area_a['x'] + ROI_SIZE_A
            ]
            area_b_roi_hsv = image_hsv[
                area_b['y'] - ROI_SIZE_B:area_b['y'] + ROI_SIZE_B,
                area_b['x'] - ROI_SIZE_B:area_b['x'] + ROI_SIZE_B
            ]
            
            # Compare ROI colors against reference shades
            print("⚙ Analyzing color patterns...")
            best_match_a, dist_a = compare_dominant_color(area_a_roi_hsv)
            best_match_b, dist_b = compare_dominant_color(area_b_roi_hsv)
            
            if not best_match_a or not best_match_b:
                return None, "Failed to match colors against reference shades.", None
            
            # Get shade numbers
            match_a_num = best_match_a['number']
            match_b_num = best_match_b['number']
            
            # Calculate color score difference
            color_score_diff = abs(match_a_num - match_b_num)
            
            # Determine B12 status based on color difference threshold
            # Difference > 1.49 indicates potential deficiency
            vitamin_b12_status = "Deficient" if color_score_diff > 1.49 else "Sufficient"
            
            print(f"✓ Area A shade: {match_a_num}, Area B shade: {match_b_num}")
            print(f"✓ Color difference: {color_score_diff:.2f}")
            print(f"✓ B12 Status: {vitamin_b12_status}")
            
            # Create visualization
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            processed_image_path = os.path.join(processed_folder, f"processed_{base_filename}.png")
            
            visualize_results(
                image_rgb, area_a, area_b, match_a_num, match_b_num,
                processed_image_path, ROI_SIZE_A, ROI_SIZE_B
            )
            
            return processed_image_path, vitamin_b12_status, color_score_diff
    
    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"✗ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error processing image: {str(e)}", None


if __name__ == "__main__":
    """
    Test the module with a sample image.
    """
    print("=" * 60)
    print("Vitamin B12 Hand Image Analysis - Test Mode")
    print("=" * 60)
    
    # Test with a sample image
    test_image_path = "path_to_hand_image.jpg"
    processed_path, status, diff = process_image(test_image_path)
    
    if processed_path is None:
        print(f"✗ Error: {status}")
    else:
        print(f"✓ Status: {status}")
        print(f"✓ Color Difference: {diff:.2f}")
        print(f"✓ Processed image saved: {processed_path}")