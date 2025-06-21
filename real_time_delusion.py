# So we will try to use the model for real-time exercise recognition.
# Since the model is trained on videos with 1 repetition, we will use a sliding window approach to capture the temporal context.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from hierarchical_transformer_prototype import HierarchicalTransformer
from hierarchical_transformer_model_v2 import HierarchicalTransformer as HierarchicalTransformerV2
from keypoint_extractor import KeypointExtractorV2
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe import solutions


class RealTimeExerciseRecognition:
    def __init__(self, model_path: str, landmarker_model: str = "models/mediapipe/pose_landmarker_full.task"):
        self.model_path = model_path
        self.keypoint_extractor = KeypointExtractorV2(model_path=landmarker_model)
        
        self.num_frames=200
        
        # Initialize the HierarchicalTransformer model
        self.model = HierarchicalTransformerV2(
            num_joints=33,
            num_frames=200, # This must match the max_frames used in training
            d_model=64,
            nhead=4,
            num_spatial_layers=1,
            num_temporal_layers=1,
            num_classes=3,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # Load the trained model weights
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'))) # map_location for flexibility
            print(f"HierarchicalTransformer weights loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model weights file not found at {self.model_path}. "
                  "Please ensure the path is correct.")
            exit() # Exit if the model cannot be loaded
        except Exception as e:
            print(f"An unexpected error occurred while loading model weights: {e}")
            exit()
            
        # Set model to evaluation mode and move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode once here

        # Get the model's expected input sequence length from its own property
        self.model_input_seq_len = self.num_frames 
        
        self.class_labels = {0: "Squats", 1: "Deadlifts", 2: "Shoulder Press"}
        
        # mediapipe setup
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = PoseLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path=landmarker_model,
                delegate=mp.tasks.BaseOptions.Delegate.CPU # Using CPU as default
            ),
            running_mode=VisionRunningMode.VIDEO
        )
        
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image
    
    # The 'padding' method is no longer needed in its original form for live inference
    # because the 'window_size' is set to match 'model_input_seq_len', and
    # the masking logic handles zeros internally.
    # def padding(self, keypoints, max_frames=200):
    #     pad_len = max_frames - len(keypoints)
    #     if pad_len > 0:
    #         pad = np.zeros((pad_len, keypoints.shape[1], keypoints.shape[2]))
    #         padded_sample = np.concatenate((keypoints, pad), axis=0)
    #     else:
    #         padded_sample = keypoints
    #     return np.array(padded_sample)
    
    def _is_prediction_stable(self, recent_predictions, confidence_threshold):
        """Check if recent predictions are stable and confident"""
        if len(recent_predictions) < 3: # Need at least 3 recent predictions to check stability
            return False
        
        # Filter for predictions with sufficient model confidence
        high_conf_predictions = [p for p in recent_predictions 
                                 if p['confidence'] >= confidence_threshold]
        
        if len(high_conf_predictions) < len(recent_predictions) * 0.5: # At least 50% should be confident
            return False
        
        if not high_conf_predictions: # Avoid error if list is empty after filtering
            return False

        # Find the most common class among high-confidence predictions
        from collections import Counter
        class_votes = Counter([p['class'] for p in high_conf_predictions])
        
        if not class_votes: # Should not happen if high_conf_predictions is not empty
            return False

        most_common_class, count = class_votes.most_common(1)[0]
        
        # Check for strong agreement: e.g., 80% of high-confidence predictions agree
        agreement_ratio = count / len(high_conf_predictions)
        
        return agreement_ratio >= 0.8 # Increased agreement threshold for stability
    
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
        status_text = "Pose: OK" if pose_detected else "Pose: WEAK/NONE" # More descriptive
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
        # Initialize video capture
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"Attempting to open video: {video_path}")
        else:
            cap = cv2.VideoCapture(0) # Use default webcam if no path provided
            print("Attempting to open webcam.")

        if not cap.isOpened():
            print("Error: Could not open video capture. Exiting.")
            return
        
        fidx = 0
        frame_window = [] # Stores keypoint data (33, 4) or (33, 4) zeros for each frame in the window
        
        # window_size MUST match model's expected sequence length
        window_size = self.model_input_seq_len 
        
        prediction_text = "Buffering frames..." # Initial status
        confidence = 0.0
        pose_confidence_threshold = 0.5 # Minimum confidence for pose landmarks (MediaPipe's internal visibility score)
        
        recent_predictions = []
        prediction_history_size = 5 # Number of recent predictions to consider for smoothing
        
        # Using a context manager for MediaPipe's PoseLandmarker
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # End of video or error reading frame
                    print("End of video or failed to read frame.")
                    break
                
                # Convert frame to MediaPipe Image format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                # Get current timestamp for video processing (more robust than fidx for video)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Extract keypoints and handle confidence
                # self.keypoint_extractor._extract_keypoints is expected to return (33, 4) or None
                current_frame_output = self.keypoint_extractor._extract_keypoints(pose_landmarker_result)
                
                pose_detected = False
                avg_landmark_confidence = 0.0
                
                if current_frame_output is not None and current_frame_output.shape == (33, 4):
                    visibility_scores = current_frame_output[:, 3]
                    avg_landmark_confidence = np.mean(visibility_scores)
                    
                    if avg_landmark_confidence >= pose_confidence_threshold:
                        frame_window.append(current_frame_output) # Append actual keypoints
                        pose_detected = True
                    else:
                        # Pose detected but low confidence - append zeros to represent 'no valid pose' for this frame
                        frame_window.append(np.zeros((33, 4), dtype=np.float32))
                else:
                    # No pose detected in the frame at all - append zeros
                    frame_window.append(np.zeros((33, 4), dtype=np.float32))
                
                # Maintain sliding window of fixed size (self.model_input_seq_len)
                if len(frame_window) > window_size:
                    frame_window.pop(0)  # Remove oldest frame
                
                # Only run inference when we have a full window of frames
                if len(frame_window) == window_size:
                    try:
                        # Prepare input tensor and attention mask from the frame_window
                        processed_frames_for_tensor = []
                        mask_values_for_tensor = []

                        for frame_data in frame_window:
                            # If a frame was replaced by zeros (low/no pose), mask it out
                            if np.array_equal(frame_data, np.zeros((33, 4), dtype=np.float32)):
                                processed_frames_for_tensor.append(np.zeros((33, 3), dtype=np.float32)) # Only XYZ, value of 0
                                mask_values_for_tensor.append(0.0) # Mask out this frame
                            else:
                                processed_frames_for_tensor.append(frame_data[:, :3]) # Take only XYZ coordinates
                                mask_values_for_tensor.append(1.0) # This frame is valid
                        
                        # Convert to PyTorch tensors and add batch dimension (batch_size=1)
                        # Input X: (1, window_size, 33, 3)
                        # Mask: (1, window_size)
                        X_tensor = torch.tensor(np.array(processed_frames_for_tensor), dtype=torch.float32).unsqueeze(0)
                        Mask_tensor = torch.tensor(np.array(mask_values_for_tensor), dtype=torch.float32).unsqueeze(0)
                        
                        # Move tensors to the correct device (CPU/GPU)
                        X_tensor = X_tensor.to(self.device)
                        Mask_tensor = Mask_tensor.to(self.device)

                        # Run inference with no gradient calculation
                        with torch.no_grad():
                            logits = self.model(X_tensor, temporal_mask=Mask_tensor)
                            
                            probabilities = torch.softmax(logits, dim=1)
                            max_prob, predicted_class_idx = torch.max(probabilities, dim=1)
                            
                            prediction_confidence = max_prob.item()
                            predicted_class = predicted_class_idx.item()
                            
                            # Track recent predictions for smoothing
                            recent_predictions.append({
                                'class': predicted_class,
                                'confidence': prediction_confidence,
                                'pose_confidence': avg_landmark_confidence
                            })
                            
                            if len(recent_predictions) > prediction_history_size:
                                recent_predictions.pop(0)
                            
                            # Determine if prediction is stable and confident enough
                            if self._is_prediction_stable(recent_predictions, confidence_threshold):
                                prediction_text = f"{self.class_labels[predicted_class]}" # Stable, just show the class
                            else:
                                # Show the current prediction, but indicate it's uncertain
                                prediction_text = f"{self.class_labels[predicted_class]} (Unstable)" 
                            
                            # Print current frame's prediction status
                            print(f"Frame {fidx}: Predicted: {self.class_labels[predicted_class]} "
                                  f"(Model Conf: {prediction_confidence:.3f}, "
                                  f"Pose Conf: {avg_landmark_confidence:.3f}, "
                                  f"Stable: {self._is_prediction_stable(recent_predictions, confidence_threshold)})")
                            
                    except Exception as e:
                        print(f"Inference error on frame {fidx}: {e}")
                        prediction_text = "Error in prediction"
                        confidence = 0.0 # Reset confidence on error

                # Draw landmarks on the original frame
                annotated_image = self.draw_landmarks_on_image(frame, pose_landmarker_result)
                
                # Overlay confidence information on the annotated image
                self._draw_confidence_overlay(annotated_image, prediction_text, confidence, 
                                                 avg_landmark_confidence, pose_detected)
                
                # Display the frame
                cv2.imshow('Real-Time Exercise Recognition', annotated_image) # Changed window title
                
                # Handle key press to exit (wait 1ms for display update)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("'q' pressed. Exiting.")
                    break
                
                fidx += 1 # Increment frame index
        
        # Release video capture and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = "models/hierarchical transformer/hierarchical_transformer_v2_weights_2025-06-21.pth"
    real_time_recognizer = RealTimeExerciseRecognition(
        model_path=model_path,
        landmarker_model="models/mediapipe/pose_landmarker_lite.task"
    )
    # real_time_recognizer.run(video_path="data/unseen/jpt.mp4")
    real_time_recognizer.run()
    # real_time_recognizer.run(video_path="data/raw/deadlifts/700_F_676330024_bp3Sa9hAVlxHyHDzXTMXrkG58zneF7aQ_ST_V1-0113.mp4")
if __name__ == "__main__":
    main()