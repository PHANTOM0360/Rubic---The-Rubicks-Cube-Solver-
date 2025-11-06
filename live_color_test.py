import cv2
import pickle
import numpy as np

# Load the trained model
try:
    model_data = pickle.load(open("rgb_model.sav", "rb"))
    model = model_data['model']
    scaler = model_data['scaler']
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Map numeric predictions to color names
COLOR_NAMES = {
    0: "green",
    1: "white",
    2: "red",
    3: "orange",
    4: "blue",
    5: "yellow"
}

# Color for display (BGR format for OpenCV)
DISPLAY_COLORS = {
    0: (0, 200, 0),      # green
    1: (255, 255, 255),  # white
    2: (0, 0, 204),      # red
    3: (51, 153, 255),   # orange
    4: (255, 90, 90),    # blue
    5: (51, 255, 255)    # yellow
}

def detect_and_classify_face(frame, model, scaler):
    """
    Detect and classify all 9 stickers on a cube face
    """
    height, width = frame.shape[:2]
    
    # Calculate grid position (centered)
    grid_size = min(width, height) * 0.6
    grid_x = int((width - grid_size) / 2)
    grid_y = int((height - grid_size) / 2)
    cell_size = int(grid_size / 3)
    
    # Sample size (area to extract color from)
    sample_size = int(cell_size * 0.3)
    
    detected_colors = []
    
    # Process each of the 9 cells
    for row in range(3):
        for col in range(3):
            # Calculate cell position
            x = grid_x + (col * cell_size)
            y = grid_y + (row * cell_size)
            
            # Calculate sample area (center of cell)
            sample_x = x + (cell_size - sample_size) // 2
            sample_y = y + (cell_size - sample_size) // 2
            
            # Extract color from sample area
            roi = frame[sample_y:sample_y + sample_size, sample_x:sample_x + sample_size]
            
            if roi.size > 0:
                # Get average BGR color
                b, g, r = cv2.mean(roi)[:3]
                
                # Predict color
                bgr_sample = np.array([[b, g, r]])
                bgr_scaled = scaler.transform(bgr_sample)
                color_id = model.predict(bgr_scaled)[0]
                probabilities = model.predict_proba(bgr_scaled)[0]
                confidence = probabilities[color_id] * 100
                
                detected_colors.append({
                    'position': (row, col),
                    'color_id': color_id,
                    'color_name': COLOR_NAMES[color_id],
                    'confidence': confidence,
                    'bgr': (int(b), int(g), int(r)),
                    'cell_coords': (x, y, cell_size),
                    'sample_coords': (sample_x, sample_y, sample_size)
                })
    
    return detected_colors

def draw_detection_grid(frame, detected_colors, show_confidence=True):
    """
    Draw the 3x3 grid with detected colors
    """
    height, width = frame.shape[:2]
    
    # Calculate grid position
    grid_size = min(width, height) * 0.6
    grid_x = int((width - grid_size) / 2)
    grid_y = int((height - grid_size) / 2)
    
    # Draw corner alignment markers
    marker_size = 25
    marker_color = (0, 255, 255)
    thickness = 3
    
    # Top-left
    cv2.line(frame, (grid_x, grid_y), (grid_x + marker_size, grid_y), marker_color, thickness)
    cv2.line(frame, (grid_x, grid_y), (grid_x, grid_y + marker_size), marker_color, thickness)
    
    # Top-right
    cv2.line(frame, (grid_x + int(grid_size), grid_y), 
             (grid_x + int(grid_size) - marker_size, grid_y), marker_color, thickness)
    cv2.line(frame, (grid_x + int(grid_size), grid_y), 
             (grid_x + int(grid_size), grid_y + marker_size), marker_color, thickness)
    
    # Bottom-left
    cv2.line(frame, (grid_x, grid_y + int(grid_size)), 
             (grid_x + marker_size, grid_y + int(grid_size)), marker_color, thickness)
    cv2.line(frame, (grid_x, grid_y + int(grid_size)), 
             (grid_x, grid_y + int(grid_size) - marker_size), marker_color, thickness)
    
    # Bottom-right
    cv2.line(frame, (grid_x + int(grid_size), grid_y + int(grid_size)), 
             (grid_x + int(grid_size) - marker_size, grid_y + int(grid_size)), marker_color, thickness)
    cv2.line(frame, (grid_x + int(grid_size), grid_y + int(grid_size)), 
             (grid_x + int(grid_size), grid_y + int(grid_size) - marker_size), marker_color, thickness)
    
    # Draw each cell
    for color_data in detected_colors:
        x, y, cell_size = color_data['cell_coords']
        sample_x, sample_y, sample_size = color_data['sample_coords']
        color_id = color_data['color_id']
        confidence = color_data['confidence']
        row, col = color_data['position']
        
        # Draw cell border (thicker for outer edges)
        border_color = (0, 255, 0) if (row == 0 or row == 2) and (col == 0 or col == 2) else (100, 100, 100)
        cv2.rectangle(frame, (x, y), (x + cell_size, y + cell_size), border_color, 2)
        
        # Draw sample area
        cv2.rectangle(frame, (sample_x, sample_y), 
                     (sample_x + sample_size, sample_y + sample_size), 
                     DISPLAY_COLORS[color_id], 2)
        
        # Fill center with detected color
        center_size = sample_size // 3
        center_x = sample_x + (sample_size - center_size) // 2
        center_y = sample_y + (sample_size - center_size) // 2
        cv2.rectangle(frame, (center_x, center_y), 
                     (center_x + center_size, center_y + center_size), 
                     DISPLAY_COLORS[color_id], -1)
        
        # Draw color label
        label = COLOR_NAMES[color_id][:3].upper()
        label_x = x + cell_size // 2 - 15
        label_y = y + cell_size // 2 + 5
        
        # Black background for text
        cv2.rectangle(frame, (label_x - 5, label_y - 20), 
                     (label_x + 35, label_y + 5), (0, 0, 0), -1)
        
        cv2.putText(frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, DISPLAY_COLORS[color_id], 2)
        
        # Show confidence if enabled
        if show_confidence and confidence < 95:
            conf_text = f"{confidence:.0f}%"
            cv2.putText(frame, conf_text, (x + 5, y + cell_size - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame

def get_cube_string(detected_colors):
    """
    Convert detected colors to cube string for solver
    """
    # Sort by position
    sorted_colors = sorted(detected_colors, key=lambda x: (x['position'][0], x['position'][1]))
    
    cube_str = ""
    for color_data in sorted_colors:
        color_id = color_data['color_id']
        if color_id == 0:
            cube_str += "F"
        elif color_id == 1:
            cube_str += "U"
        elif color_id == 2:
            cube_str += "R"
        elif color_id == 3:
            cube_str += "L"
        elif color_id == 4:
            cube_str += "B"
        elif color_id == 5:
            cube_str += "D"
    
    return cube_str

# Main program
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera!")
    exit()

print("\n" + "="*60)
print("🎲 FULL RUBIK'S CUBE FACE DETECTION (9 Colors)")
print("="*60)
print("Instructions:")
print("- Align the cube face with the cyan corner markers")
print("- Make sure all 9 stickers are visible in the grid")
print("- Press 'SPACE' to capture and freeze detection")
print("- Press 'c' to continue scanning")
print("- Press 's' to toggle confidence display")
print("- Press 'q' to quit")
print("="*60 + "\n")

frozen_detection = None
frozen_printed = False  # FIXED: Track if we've printed the frozen detection
show_confidence = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frozen_detection is None:
        # Live detection mode
        detected_colors = detect_and_classify_face(frame, model, scaler)
        frozen_printed = False  # Reset printed flag when not frozen
        
        if len(detected_colors) == 9:
            frame = draw_detection_grid(frame, detected_colors, show_confidence)
            
            # Show instructions
            cv2.putText(frame, "Press SPACE to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show center sticker prominently
            center_color = detected_colors[4]  # Middle sticker
            center_text = f"Center: {center_color['color_name'].upper()}"
            cv2.rectangle(frame, (10, 50), (280, 85), (0, 0, 0), -1)
            cv2.putText(frame, center_text, (15, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       DISPLAY_COLORS[center_color['color_id']], 2)
        else:
            cv2.putText(frame, f"Detecting... ({len(detected_colors)}/9)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # FIXED: frozen_detection is now always a list, not a string
        # Frozen mode - show captured detection
        frame = draw_detection_grid(frame, frozen_detection, show_confidence)
        
        # Show cube string
        cube_str = get_cube_string(frozen_detection)
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame, "CAPTURED!", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"String: {cube_str}", (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'c' to continue", (15, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Print to console only once
        if not frozen_printed:
            center_color = frozen_detection[4]
            print(f"\n📊 Detected Face (Center: {center_color['color_name'].upper()}):")
            print(f"Cube String: {cube_str}")
            print("\nGrid Layout:")
            for i in range(3):
                row_colors = []
                for j in range(3):
                    idx = i * 3 + j
                    color_name = frozen_detection[idx]['color_name']
                    confidence = frozen_detection[idx]['confidence']
                    row_colors.append(f"{color_name[:3].upper()}({confidence:.0f}%)")
                print("  ".join(row_colors))
            frozen_printed = True  # Mark as printed

    cv2.imshow("Full Cube Face Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # Space to capture
        if frozen_detection is None and len(detected_colors) == 9:
            frozen_detection = detected_colors
            frozen_printed = False  # Reset to print on next frame
            print("\n✅ Face captured!")
    
    elif key == ord('c'):  # Continue scanning
        frozen_detection = None
        frozen_printed = False
        print("\n🔄 Continuing live detection...")
    
    elif key == ord('s'):  # Toggle confidence display
        show_confidence = not show_confidence
        print(f"\n{'✅' if show_confidence else '❌'} Confidence display: {show_confidence}")
    
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection completed.")