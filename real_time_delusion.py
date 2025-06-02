# So we will try to use the model for real-time exercise recognition.
# Since the model is trained on videos with 1 repetition, we will use a sliding window approach to capture the temporal context.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from hierarchical_transformer_prototype import HierarchicalTransformer
from keypoint_extractor import KeypointExtractorV2
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe import solutions


class RealTimeExerciseRecognition:
    def __init__(self, model_path: str, landmarker_model: str = "models/mediapipe/pose_landmarker_full.task"):
        self.model_path = model_path
        self.keypoint_extractor = KeypointExtractorV2(model_path=landmarker_model)
        self.model = HierarchicalTransformer(
            num_joints=33,
            num_frames=331,
            d_model=64,
            nhead=4,
            num_spatial_layers=1,
            num_temporal_layers=1,
            num_classes=3,
            dim_feedforward=512
        )
        self.model.load_state_dict(torch.load(self.model_path))
        
        self.class_labels = {0: "Squats", 1: "Deadlifts", 2: "Shoulder Press"}
        
        
        # mediapipe shit
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = PoseLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path=landmarker_model,
                delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.VIDEO
        )
        
        
    def draw_landmarks_on_image(self, rgb_image, detection_result):
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
    
    
    def padding(self, keypoints):
        max_frames = 331
        pad_len = max_frames - len(keypoints)
        if pad_len > 0:
            pad = np.zeros((pad_len, keypoints.shape[1], keypoints.shape[2]))  # Preserve all dimensions
            padded_sample = np.concatenate((keypoints, pad), axis=0)
        else:
            padded_sample = keypoints

        return np.array(padded_sample)
    
    def run(self, video_path:str = None):
        v = 0
        if video_path:
            v=video_path
            
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print("Error: Could not open video capture.")
            return
        
        fidx = 0
        frame_window = []
        window_size = 30  # Adjust based on your model's expected sequence length
        prediction_text = "Waiting for poses..."
        confidence = 0.0
        
        
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                pose_landmarker_result = landmarker.detect_for_video(mp_image, fidx)
                
                current_frame_output = self.keypoint_extractor._extract_keypoints(pose_landmarker_result)
                
                # Add current frame to window (with zero padding if no pose detected)
                if current_frame_output is not None and len(current_frame_output) == 33:
                    frame_window.append(current_frame_output)
                else:
                    # No pose detected or incomplete pose - add zeros
                    frame_window.append(np.zeros((33, 4)))
                    
                    
                # Maintain sliding window of fixed size
                if len(frame_window) > window_size:
                    frame_window.pop(0)  # Remove oldest frame
                
                # Run inference when we have enough frames
                if len(frame_window) == window_size:
                    try:
                        # Convert to model input format
                        np_frame_window = np.array(frame_window)  # Shape: (window_size, 33, 4)
                        padded_sequence = self.padding(np_frame_window)  # Shape: (window_size, 33, 4) ouput: (331, 33, 4)
                        
                        X_sample = padded_sequence[:, :, :3] # removes the visibility score
                        x_tensor = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0)
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.model(x_tensor)
                            predicted_class = torch.argmax(logits, dim=1).item()
                            print(f"Predicted class: {self.class_labels[predicted_class]}")
                            
                    except Exception as e:
                        print(f"Inference error: {e}")

                # for display purposes, we will draw the landmarks on the image
                annotated_image = self.draw_landmarks_on_image(frame, pose_landmarker_result) 
                cv2.imshow('frame', annotated_image)
                
                # Fix: Need to capture the key press properly
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                fidx += 1
        
        cap.release()
        cv2.destroyAllWindows()
            

def main():
    model_path = "models/hierarchical transformer/hierarchical_transformer_weights_2025-06-03_small_1.pth"
    real_time_recognizer = RealTimeExerciseRecognition(
        model_path=model_path,
        landmarker_model="models/mediapipe/pose_landmarker_lite.task"
    )
    real_time_recognizer.run()
    
if __name__ == "__main__":
    main()