import cv2
import pickle
import numpy as np

# Load the trained model
try:
    model = pickle.load(open("rgb_model.sav", "rb"))
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: rgb_model.sav not found!")
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

cap = cv2.VideoCapture(0)
print("\n🎥 Rubik's Cube Color Detection Active")
print("Align a sticker inside the green square.\nPress 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2
    size = 40

    # Draw detection box
    cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 2)

    # Extract center region
    roi = frame[cy - size:cy + size, cx - size:cx + size]
    b, g, r = cv2.mean(roi)[:3]
    rgb = np.array([[r, g, b]])

    # Predict color using model
    color_id = model.predict(rgb)[0]
    color_name = COLOR_NAMES.get(color_id, "unknown")
    display_color = DISPLAY_COLORS.get(color_id, (255, 255, 255))

    # Display predicted color with colored background
    text = f"Color: {color_name.upper()}"
    
    # Add background rectangle for text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    cv2.rectangle(frame, (5, 5), (text_size[0] + 15, 50), (0, 0, 0), -1)
    
    # Put text
    cv2.putText(frame, text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, display_color, 2)
    
    # Show RGB values for debugging
    cv2.putText(frame, f"RGB: ({int(r)}, {int(g)}, {int(b)})", 
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Rubik's Cube Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🟢 Detection Ended.")