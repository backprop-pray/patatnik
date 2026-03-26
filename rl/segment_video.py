import cv2
import numpy as np
import ssl
import sys
from ultralytics import FastSAM

# Fix for macOS Python SSL certificate download issues
ssl._create_default_https_context = ssl._create_unverified_context

def process_video(input_path, output_path, mask_only=False):
    print(f"Loading FastSAM model...")
    model = FastSAM("FastSAM-s.pt")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # We will stack Original + Mask side-by-side, so double the width
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    print(f"Processing {total_frames} frames from {input_path}...")
    print(f"Output (Side-by-Side) will be saved to {output_path}")
    print("Press 'q' to stop early.")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame, verbose=False, imgsz=256)
        
        # 1. Prepare Mask Frame (Black background)
        blank_canvas = np.zeros_like(frame)
        mask_frame = results[0].plot(img=blank_canvas, labels=False, boxes=False)

        # 2. Combine Side-by-Side: [Original | Mask]
        combined_frame = np.hstack((frame, mask_frame))

        # Write the combined frame
        out.write(combined_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            sys.stdout.write(f"\rProgress: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
            sys.stdout.flush()

        # Display
        cv2.imshow('Original vs Mask Segmentation', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished! Side-by-side video saved to {output_path}")

if __name__ == "__main__":
    # Update these paths to the video you want to process
    INPUT_FILE = "raw_farm_video.mp4"
    OUTPUT_FILE = "segmented_farm_comparison.mp4"
    
    process_video(INPUT_FILE, OUTPUT_FILE)
