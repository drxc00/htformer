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
    
    def _is_prediction_stable(self, recent_predictions, confidence_threshold):
        """Check if recent predictions are stable and confident"""
        if len(recent_predictions) < 3:
            return False
        
        # Check if recent predictions have sufficient confidence
        high_conf_predictions = [p for p in recent_predictions 
                               if p['confidence'] >= confidence_threshold]
        
        if len(high_conf_predictions) < 2:
            return False
        
        # Check if the most confident predictions agree
        most_common_class = max(set(p['class'] for p in high_conf_predictions),
                              key=[p['class'] for p in high_conf_predictions].count)
        
        # Count how many recent high-confidence predictions match the most common
        matching_predictions = sum(1 for p in high_conf_predictions 
                                 if p['class'] == most_common_class)
        
        return matching_predictions >= len(high_conf_predictions) * 0.6  # 60% agreement

    def _draw_confidence_overlay(self, image, prediction_text, model_confidence, 
                               pose_confidence, pose_detected):
        """Draw confidence information on the image"""
        height, width = image.shape[:2]
        
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Prediction text
        cv2.putText(image, f"Exercise: {prediction_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Model confidence
        conf_color = (0, 255, 0) if model_confidence > 0.7 else (0, 255, 255) if model_confidence > 0.5 else (0, 0, 255)
        cv2.putText(image, f"Model Conf: {model_confidence:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        # Pose detection confidence
        pose_color = (0, 255, 0) if pose_confidence > 0.7 else (0, 255, 255) if pose_confidence > 0.5 else (0, 0, 255)
        cv2.putText(image, f"Pose Conf: {pose_confidence:.2f}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        # Pose detection status
        status_text = "Pose: OK" if pose_detected else "Pose: WEAK"
        status_color = (0, 255, 0) if pose_detected else (0, 0, 255)
        cv2.putText(image, status_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Confidence bars
        self._draw_confidence_bar(image, width - 150, 30, model_confidence, "Model")
        self._draw_confidence_bar(image, width - 150, 60, pose_confidence, "Pose")

    def _draw_confidence_bar(self, image, x, y, confidence, label):
        """Draw a confidence bar"""
        bar_width = 100
        bar_height = 15
        
        # Background bar
        cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + conf_width, y + bar_height), color, -1)
        
        # Label
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self, video_path: str = None, confidence_threshold: float = 0.7):
        v = 0
        if video_path:
            v = video_path
            
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print("Error: Could not open video capture.")
            return
        
        fidx = 0
        frame_window = []
        window_size = 60  # Adjust based on your model's expected sequence length
        prediction_text = "Waiting for poses..."
        confidence = 0.0
        pose_confidence_threshold = 0.5  # Minimum confidence for pose landmarks
        
        # Track prediction stability
        recent_predictions = []
        prediction_history_size = 5
        
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                pose_landmarker_result = landmarker.detect_for_video(mp_image, fidx)
                
                # Extract keypoints with confidence filtering
                current_frame_output = self.keypoint_extractor._extract_keypoints(pose_landmarker_result)
                
                # Check pose detection confidence
                pose_detected = False
                avg_landmark_confidence = 0.0
                
                if current_frame_output is not None and len(current_frame_output) == 33:
                    # Calculate average confidence from visibility scores (4th column)
                    visibility_scores = current_frame_output[:, 3]
                    avg_landmark_confidence = np.mean(visibility_scores)
                    
                    # Only use pose if confidence is above threshold
                    if avg_landmark_confidence >= pose_confidence_threshold:
                        frame_window.append(current_frame_output)
                        pose_detected = True
                    else:
                        # Low confidence pose - treat as no pose
                        frame_window.append(np.zeros((33, 4)))
                else:
                    # No pose detected
                    frame_window.append(np.zeros((33, 4)))
                
                # Maintain sliding window of fixed size
                if len(frame_window) > window_size:
                    frame_window.pop(0)  # Remove oldest frame
                
                # Run inference when we have enough frames
                if len(frame_window) == window_size:
                    try:
                        # Convert to model input format
                        np_frame_window = np.array(frame_window)  # Shape: (window_size, 33, 4)
                        padded_sequence = self.padding(np_frame_window)  # Shape: (331, 33, 4)
                        
                        X_sample = padded_sequence[:, :, :3]  # removes the visibility score
                        x_tensor = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0)
                        
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.model(x_tensor)
                            
                            # Get prediction confidence using softmax
                            probabilities = torch.softmax(logits, dim=1)
                            max_prob, predicted_class = torch.max(probabilities, dim=1)
                            
                            prediction_confidence = max_prob.item()
                            predicted_class = predicted_class.item()
                            
                            # Track recent predictions for stability
                            recent_predictions.append({
                                'class': predicted_class,
                                'confidence': prediction_confidence,
                                'pose_confidence': avg_landmark_confidence
                            })
                            
                            if len(recent_predictions) > prediction_history_size:
                                recent_predictions.pop(0)
                            
                            # Determine if prediction is stable and confident
                            if self._is_prediction_stable(recent_predictions, confidence_threshold):
                                prediction_text = f"{self.class_labels[predicted_class]}"
                                confidence = prediction_confidence
                            else:
                                prediction_text = "Uncertain..."
                                confidence = prediction_confidence
                            
                            print(f"Predicted: {self.class_labels[predicted_class]} "
                                  f"(Model Conf: {prediction_confidence:.3f}, "
                                  f"Pose Conf: {avg_landmark_confidence:.3f})")
                            
                    except Exception as e:
                        print(f"Inference error: {e}")
                        prediction_text = "Error in prediction"
                        confidence = 0.0

                # Draw landmarks and overlay confidence information
                annotated_image = self.draw_landmarks_on_image(frame, pose_landmarker_result)
                
                # Add confidence visualization
                self._draw_confidence_overlay(annotated_image, prediction_text, confidence, 
                                            avg_landmark_confidence, pose_detected)
                
                cv2.imshow('frame', annotated_image)
                
                # Fix: Need to capture the key press properly
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                fidx += 1
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = "models/hierarchical transformer/hierarchical_transformer_weights_2025-06-03_small_2.pth"
    real_time_recognizer = RealTimeExerciseRecognition(
        model_path=model_path,
        landmarker_model="models/mediapipe/pose_landmarker_lite.task"
    )
    # real_time_recognizer.run(video_path="data/unseen/jpt.mp4")
    real_time_recognizer.run()
if __name__ == "__main__":
    main()