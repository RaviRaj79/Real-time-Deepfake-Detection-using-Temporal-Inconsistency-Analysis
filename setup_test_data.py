import cv2
import numpy as np
import os

def create_test_videos():
    """Create valid test video files for the project."""
    
    # Create deepfake and video folders if they don't exist
    os.makedirs("deepfake", exist_ok=True)
    os.makedirs("video", exist_ok=True)
    os.makedirs("image", exist_ok=True)
    
    # Video parameters
    fps = 30
    width, height = 320, 240
    duration = 2  # 2 seconds
    num_frames = fps * duration
    
    # Create deepfake videos
    for i in range(1, 6):
        video_path = f"deepfake/{i}.mp4" if i <= 4 else f"deepfake/{i}.mov"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame_num in range(num_frames):
            # Create animated frames with color changes (simulating deepfake artifacts)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            hue = int((frame_num / num_frames) * 180)
            
            # Create a pattern that changes over time
            for y in range(height):
                for x in range(width):
                    frame[y, x] = [hue, 255, 255]
            
            # Convert HSV to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Created: {video_path}")
    
    # Create real videos
    for i in range(1, 6):
        video_path = f"video/{i}.mp4" if i <= 4 else f"video/{i}.mov"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame_num in range(num_frames):
            # Create animated frames with less variation (normal video)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Simpler pattern for real videos
            intensity = int((frame_num / num_frames) * 255)
            frame[:, :] = [intensity, intensity, intensity]
            out.write(frame)
        
        out.release()
        print(f"Created: {video_path}")

def create_test_images():
    """Create valid test image files for the project."""
    
    os.makedirs("image", exist_ok=True)
    
    # Create test images
    for i in range(1, 6):
        img_name = f"image/{i}.jpg" if i != 2 else f"image/{i}.jpeg"
        
        # Create a simple colored image
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        color = (50 + i * 40, 100 + i * 30, 150 + i * 20)
        img[:, :] = color
        
        # Add some pattern
        cv2.circle(img, (160, 120), 50 + i * 10, (255, 255, 255), -1)
        
        cv2.imwrite(img_name, img)
        print(f"Created: {img_name}")

if __name__ == "__main__":
    print("Creating test video files...")
    create_test_videos()
    print("\nCreating test image files...")
    create_test_images()
    print("\nAll test files created successfully!")
