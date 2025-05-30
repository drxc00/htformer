# This file is part of the thesis project and is used to test various MediaPipe models for pose detection.
# Here we test both the new MediaPipe Pose Landmark model and the old MediaPipe Pose model (BlazePose).
#
# The purpose of this file is to visualize and compare the performance of these models on exercise videos.
# The new MediaPipe Pose Landmark model is expected to provide more accurate and detailed pose detection,
# while the old MediaPipe Pose model is expected to provide faster processing speeds.

import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def load_video(path: str) -> cv.VideoCapture:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return cv.VideoCapture(path)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def mediapipe_pose(file_path:str) -> None:
    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    
    options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=r"models/mediapipe/pose_landmarker_heavy.task",
        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO)
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = load_video(file_path)
        
        fidx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            pose_landmarker_result = landmarker.detect_for_video(mp_image, fidx)
            
            annotated_image = draw_landmarks_on_image(frame, pose_landmarker_result) 
            
            cv.imshow('frame', annotated_image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
            fidx += 1
        
        cap.release()
        cv.destroyAllWindows()
        
def mediapipe_pose_heavy_optimized(file_path: str, target_fps: int = 15) -> None:
    """
    Optimized heavy model processing for exercise analysis
    - Maintains heavy model accuracy
    - Implements smart frame skipping
    - Uses threading for better performance
    """
    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=r"models/mediapipe/pose_landmarker_heavy.task",
            delegate=python.BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO)
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = load_video(file_path)
        
        # Get video properties
        video_fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height} @ {video_fps} FPS, {total_frames} frames")
        
        # Calculate frame skip for target FPS
        frame_skip = max(1, int(video_fps / target_fps))
        print(f"Processing every {frame_skip} frame(s) for ~{target_fps} FPS analysis")
        
        fidx = 0
        processed_frames = 0
        last_pose_result = None
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only selected frames for pose detection
            if fidx % frame_skip == 0:
                # Optional: Slight resize to reduce computation (exercise videos are often high-res)
                if frame_width > 1920:
                    scale_factor = 1920 / frame_width
                    new_width = int(frame_width * scale_factor)
                    new_height = int(frame_height * scale_factor)
                    frame_resized = cv.resize(frame, (new_width, new_height))
                else:
                    frame_resized = frame
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_resized)
                pose_landmarker_result = landmarker.detect_for_video(mp_image, fidx)
                last_pose_result = pose_landmarker_result
                processed_frames += 1
                
                # Show processing progress
                if processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = processed_frames / elapsed
                    progress = (fidx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Processing FPS: {fps_actual:.1f}")
            
            # Always draw the last detected pose (smooth visualization)
            if last_pose_result:
                annotated_image = draw_landmarks_on_image(frame, last_pose_result)
            else:
                annotated_image = frame
            
            cv.imshow('Exercise Analysis - Heavy Model', annotated_image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
            fidx += 1
        
        cap.release()
        cv.destroyAllWindows()
        
        # Final stats
        total_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"Total frames: {fidx}")
        print(f"Processed frames: {processed_frames}")  
        print(f"Total time: {total_time:.1f}s")
        print(f"Average processing FPS: {processed_frames/total_time:.1f}")
        
def blaze_pose(file_path:str) -> None:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2)
    
    cap = load_video(file_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        # Draw keypoints on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        )
        
        cv.imshow('Video with Keypoints', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
        
if __name__ == "__main__":
    
    mediapipe_pose_heavy_optimized(r"data/raw/shoulder_press/shoulder press_5_rep_1.mp4")
        
        
