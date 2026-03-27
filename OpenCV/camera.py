import cv2
import sys

def render_live_feed(camera_index=0):
    """
    Renders live camera feed using OpenCV.
    
    Args:
        camera_index (int): The index of the camera device to use.
    """
    # Initialize the video capture object
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return
    
    print("Starting live camera feed. Press 'q' to exit.")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame.")
                break
                
            # Display the resulting frame
            cv2.imshow('Live Camera Feed', frame)
            
            # Press 'q' on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        print("Camera feed stopped.")

if __name__ == "__main__":
    # Allow camera index to be passed as a command line argument
    cam_idx = 0
    if len(sys.argv) > 1:
        try:
            cam_idx = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index provided: {sys.argv[1]}. Using default (0).")
            
    render_live_feed(cam_idx)
