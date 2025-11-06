import cv2
import pandas as pd
import os
import numpy as np

# Define all cube colors
cube_colors = ["green", "white", "red", "orange", "blue", "yellow"]

# Create output directory
os.makedirs("training_data", exist_ok=True)

cap = cv2.VideoCapture(0)

print("""
==============================
IMPROVED RUBIK'S CUBE COLOR TRAINING
==============================
Instructions:
- Hold the cube sticker for the prompted color in front of your webcam.
- Align the sticker inside the GREEN SQUARE.
- Press 'c' to capture a sample.
- Try to capture samples under different angles and lighting!
- Collect at least 20-30 samples per color for best results.
- Press 'n' to move to the next color.
- Press 'q' anytime to quit early.

IMPORTANT: For RED and ORANGE, make sure to:
- Capture samples from different angles
- Vary the lighting slightly
- Get at least 30 samples of each
==============================
""")

for color in cube_colors:
    data = []
    print(f"\n🎨 Collecting samples for {color.upper()}...")
    print(f"Current samples: 0")
    print("Press 'c' to capture, 'n' for next color, 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        cx, cy = width // 2, height // 2
        size = 50  # Larger sampling area

        # Draw main sampling box
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 3)
        
        # Draw inner sampling area (where we actually sample from)
        inner_size = 30
        cv2.rectangle(frame, (cx - inner_size, cy - inner_size), 
                     (cx + inner_size, cy + inner_size), (0, 255, 255), 2)
        
        # Display current color and sample count
        cv2.putText(frame, f"Color: {color.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Samples: {len(data)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show current RGB values
        roi = frame[cy - inner_size:cy + inner_size, cx - inner_size:cx + inner_size]
        if roi.size > 0:
            b, g, r = cv2.mean(roi)[:3]
            cv2.putText(frame, f"RGB: ({int(r)}, {int(g)}, {int(b)})", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show a preview of the sampled color
            color_preview = np.zeros((60, 200, 3), dtype=np.uint8)
            color_preview[:, :] = (b, g, r)
            frame[height - 80:height - 20, 10:210] = color_preview

        cv2.imshow("Rubik's Cube Color Trainer", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Capture from the inner sampling area
            roi = frame[cy - inner_size:cy + inner_size, cx - inner_size:cx + inner_size]
            
            if roi.size > 0:
                # Get mean color from ROI
                b, g, r = cv2.mean(roi)[:3]
                
                # Also sample from slightly offset positions for more variety
                offsets = [0, -5, 5, -10, 10]
                for offset_x in offsets:
                    for offset_y in offsets:
                        ox = cx + offset_x
                        oy = cy + offset_y
                        sample_roi = frame[oy - 5:oy + 5, ox - 5:ox + 5]
                        if sample_roi.size > 0:
                            b_s, g_s, r_s = cv2.mean(sample_roi)[:3]
                            data.append({"R": int(r_s), "G": int(g_s), "B": int(b_s)})
                
                # Add main sample
                data.append({"R": int(r), "G": int(g), "B": int(b)})
                
                print(f"Captured sample {len(data)}: R={int(r)}, G={int(g)}, B={int(b)}")

        elif key == ord('n'):
            if len(data) < 10:
                print(f"⚠️ Warning: Only {len(data)} samples collected for {color}!")
                print("Press 'n' again to confirm, or 'c' to collect more samples.")
                if cv2.waitKey(1000) & 0xFF == ord('n'):
                    break
            else:
                print(f"✅ Finished collecting {len(data)} samples for {color}.")
                if len(data) > 0:
                    df = pd.DataFrame(data)
                    df.to_excel(f"training_data/{color}.xlsx", index=False)
                break

        elif key == ord('q'):
            print("⚠️ Training aborted early.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("\n✅ All colors collected successfully!")
cap.release()
cv2.destroyAllWindows()

# Show summary
print("\n" + "="*60)
print("TRAINING DATA SUMMARY")
print("="*60)
for color in cube_colors:
    file_path = f"training_data/{color}.xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        print(f"{color.upper()}: {len(df)} samples")
print("="*60)
print("\nNow run: python color_train.py")