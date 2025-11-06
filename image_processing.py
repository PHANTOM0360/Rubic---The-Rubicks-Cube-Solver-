import cv2
import numpy as np
import pickle

# Load the trained model
try:
    loaded_model = pickle.load(open("rgb_model.sav", 'rb'))
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: rgb_model.sav not found. Please train the model first.")
    loaded_model = None

# Color mapping
COLOR_MAP = {
    "green": 0,   # F
    "white": 1,   # U
    "red": 2,     # R
    "orange": 3,  # L
    "blue": 4,    # B
    "yellow": 5   # D
}

# Reverse mapping for debugging
COLOR_NAMES = {v: k for k, v in COLOR_MAP.items()}


def detect_grid(image):
    """
    Draw a fixed 3x3 grid overlay and sample colors from each cell
    User must align the cube face with the grid
    """
    height, width = image.shape[:2]
    
    # Calculate grid position (centered)
    grid_size = min(width, height) * 0.5  # Grid takes 50% of smaller dimension
    grid_x = int((width - grid_size) / 2)
    grid_y = int((height - grid_size) / 2)
    cell_size = int(grid_size / 3)
    
    # Sample size (smaller square in center of each cell for color sampling)
    sample_size = int(cell_size * 0.4)
    
    grid = []
    
    # Draw 3x3 grid and sample colors
    for row in range(3):
        for col in range(3):
            # Calculate cell position
            x = grid_x + (col * cell_size)
            y = grid_y + (row * cell_size)
            
            # Draw cell border (green for outer border, gray for inner)
            color = (0, 255, 0) if (row == 0 or row == 2) and (col == 0 or col == 2) else (100, 100, 100)
            cv2.rectangle(image, (x, y), (x + cell_size, y + cell_size), color, 2)
            
            # Calculate sample area (center of cell)
            sample_x = x + (cell_size - sample_size) // 2
            sample_y = y + (cell_size - sample_size) // 2
            
            # Draw sample area
            cv2.rectangle(image, (sample_x, sample_y), 
                         (sample_x + sample_size, sample_y + sample_size), 
                         (255, 255, 0), 2)
            
            # Extract color from sample area
            roi = image[sample_y:sample_y + sample_size, sample_x:sample_x + sample_size]
            
            if roi.size > 0:
                b, g, r = cv2.mean(roi)[:3]
                
                # Draw center dot
                center_x = sample_x + sample_size // 2
                center_y = sample_y + sample_size // 2
                cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)
                
                # Store color data [B, G, R, position]
                # Position value maintains grid order (row-major order)
                position = row * 3 + col
                grid.append([b, g, r, position])
    
    # Add instructions
    cv2.putText(image, "Align cube face with grid", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, "Press scan button when ready", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw corner markers for better alignment
    marker_size = 20
    marker_color = (0, 255, 255)
    # Top-left
    cv2.line(image, (grid_x, grid_y), (grid_x + marker_size, grid_y), marker_color, 3)
    cv2.line(image, (grid_x, grid_y), (grid_x, grid_y + marker_size), marker_color, 3)
    # Top-right
    cv2.line(image, (grid_x + int(grid_size), grid_y), 
             (grid_x + int(grid_size) - marker_size, grid_y), marker_color, 3)
    cv2.line(image, (grid_x + int(grid_size), grid_y), 
             (grid_x + int(grid_size), grid_y + marker_size), marker_color, 3)
    # Bottom-left
    cv2.line(image, (grid_x, grid_y + int(grid_size)), 
             (grid_x + marker_size, grid_y + int(grid_size)), marker_color, 3)
    cv2.line(image, (grid_x, grid_y + int(grid_size)), 
             (grid_x, grid_y + int(grid_size) - marker_size), marker_color, 3)
    # Bottom-right
    cv2.line(image, (grid_x + int(grid_size), grid_y + int(grid_size)), 
             (grid_x + int(grid_size) - marker_size, grid_y + int(grid_size)), marker_color, 3)
    cv2.line(image, (grid_x + int(grid_size), grid_y + int(grid_size)), 
             (grid_x + int(grid_size), grid_y + int(grid_size) - marker_size), marker_color, 3)
    
    if len(grid) > 0:
        grid = np.asarray(grid)
        # Sort by position to maintain proper order
        grid = grid[grid[:, -1].argsort()]
    
    return image, grid


def classifiy_grid(grid):
    """
    Classify the detected grid colors using the trained model
    
    Args:
        grid: numpy array with shape (9, 4) containing [B, G, R, position] for each sticker
    
    Returns:
        cube_str: String representation for kociemba solver (e.g., "FFFFFFFFF")
        predictions: numpy array of numeric color predictions (0-5)
    """
    cube_str = ""
    predictions = []
    
    if loaded_model is None:
        print("❌ Model not loaded!")
        return cube_str, np.array(predictions)
    
    if len(grid) != 9:
        print(f"⚠️ Warning: Expected 9 stickers, got {len(grid)}")
        return cube_str, np.array(predictions)
    
    # Extract BGR values and convert to RGB for model
    colors_bgr = grid[:, 0:3]
    colors_rgb = colors_bgr[:, [2, 1, 0]]  # Convert BGR to RGB
    
    # Debug: Print the RGB values being classified
    print(f"\n🔍 Classifying {len(colors_rgb)} stickers:")
    for i, rgb in enumerate(colors_rgb):
        print(f"  Sticker {i+1}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
    
    # Get predictions from model
    try:
        predicted_labels = loaded_model.predict(colors_rgb)
        
        # Convert to numeric if needed
        numeric_predictions = []
        for label in predicted_labels:
            if isinstance(label, str):
                numeric_predictions.append(COLOR_MAP.get(label, 0))
            else:
                numeric_predictions.append(int(label))
        
        predictions = np.array(numeric_predictions)
        
        # Build string representation for kociemba solver
        for pred in predictions:
            if pred == 0:
                cube_str += "F"
            elif pred == 1:
                cube_str += "U"
            elif pred == 2:
                cube_str += "R"
            elif pred == 3:
                cube_str += "L"
            elif pred == 4:
                cube_str += "B"
            elif pred == 5:
                cube_str += "D"
        
        # Debug output
        color_names = [COLOR_NAMES.get(p, 'unknown') for p in predictions]
        print(f"✅ Detected colors: {color_names}")
        print(f"✅ Cube string: {cube_str}")
        if len(predictions) > 4:
            print(f"✅ Center sticker (index 4): {COLOR_NAMES.get(predictions[4], 'unknown')} (value: {predictions[4]})")
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        return "", np.array([])
    
    return cube_str, predictions