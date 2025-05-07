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
        
def blaze_pose(file_path:str) -> None:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
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
    
    mediapipe_pose(r"data/augmented/squats/32995_3_original.mp4")
        
        
