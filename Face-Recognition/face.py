import cv2
import os
import time

def record_video(output_path: str, duration_seconds: int = 5):
    print(f"\n--- Starting Video Recording (Max {duration_seconds}s) ---")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Check connectivity and permissions.")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps <= 0:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    end_time = start_time + duration_seconds
    
    print(f"Recording at {frame_width}x{frame_height} @ {fps} FPS...")
    
    while time.time() < end_time:
        ret, frame = cap.read()
        
        if ret:
            cv2.imshow('Recording', frame)
            out.write(frame)
        else:
            print("Error: Failed to read frame from webcam.")
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Successfully recorded video saved to: {output_path}")
        return True
    else:
        print(f"Error: Video file was not successfully saved or is empty.")
        return False


def extract_frames(video_path: str, output_dir: str):
    print(f"\n--- Starting Frame Extraction from {video_path} ---")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}. Cannot extract frames.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        print("Error: Could not determine video FPS. Cannot calculate time-based frame limit.")
        cap.release()
        return

    frames_to_process = total_frames
    print(f"Video Info:")
    print(f"  FPS: {fps}")
    print(f"  Total Frames in recorded video: {total_frames}")
    print(f"  Extracting {frames_to_process} frames.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/exists at: {output_dir}")

    frame_count = 0
    while frame_count < frames_to_process:
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Reached end of video or failed to read frame {frame_count}. Stopping.")
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nFinished extraction.")
    print(f"Successfully saved {frame_count} frames to '{output_dir}'.")


if __name__ == "__main__":
    RECORDED_VIDEO_FILE = 'recorded_face_data.mp4'
    OUTPUT_FOLDER = 'faces'
    MAX_DURATION = 15

    if record_video(output_path=RECORDED_VIDEO_FILE, duration_seconds=MAX_DURATION):
        extract_frames(
            video_path=RECORDED_VIDEO_FILE,
            output_dir=OUTPUT_FOLDER,
        )
        
        os.remove(RECORDED_VIDEO_FILE)
        print(f"\nCleaned up recorded video file: {RECORDED_VIDEO_FILE}")
    else:
        print("\nProcess aborted because video recording failed.")
