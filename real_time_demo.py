# So we will try to use the model for real-time exercise recognition.
# Since the model is trained on videos with 1 repetition, we will use a sliding window approach to capture the temporal context.
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import torch
from core.models.hierarchical_transformer import HierarchicalTransformer
from core.keypoint_extractor import KeypointExtractorV2
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe import solutions


class RealTimeExerciseRecognition:
    def __init__(self, model_path: str, landmarker_model: str = "models/mediapipe/pose_landmarker_full.task"):
        self.model_path = model_path
        self.keypoint_extractor = KeypointExtractorV2(
            model_path=landmarker_model)

        self.num_frames = 201

        # Initialize the HierarchicalTransformer model
        self.model = HierarchicalTransformer(
            num_joints=33,
            num_frames=201,
            d_model=64,
            nhead=2,
            num_spatial_layers=1,
            num_temporal_layers=1,
            num_classes=3,
            dim_feedforward=2048,
            dropout=0.1
        )

        # Load the trained model weights
        try:
            self.model.load_state_dict(torch.load(
                # map_location for flexibility
                self.model_path, map_location=torch.device('cpu')))
            print(
                f"HierarchicalTransformer weights loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model weights file not found at {self.model_path}. "
                  "Please ensure the path is correct.")
            exit()  # Exit if the model cannot be loaded
        except Exception as e:
            print(
                f"An unexpected error occurred while loading model weights: {e}")
            exit()

        # Set model to evaluation mode and move to device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode once here

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
                delegate=mp.tasks.BaseOptions.Delegate.CPU  # Using CPU as default
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

    def _is_prediction_stable(self, recent_predictions, confidence_threshold):
        """Check if recent predictions are stable and confident"""
        if len(recent_predictions) < 3:  # Need at least 3 recent predictions to check stability
            return False

        # Filter for predictions with sufficient model confidence
        high_conf_predictions = [p for p in recent_predictions
                                 if p['confidence'] >= confidence_threshold]

        # At least 50% should be confident
        if len(high_conf_predictions) < len(recent_predictions) * 0.5:
            return False

        if not high_conf_predictions:  # Avoid error if list is empty after filtering
            return False

        # Find the most common class among high-confidence predictions
        from collections import Counter
        class_votes = Counter([p['class'] for p in high_conf_predictions])

        if not class_votes:  # Should not happen if high_conf_predictions is not empty
            return False

        most_common_class, count = class_votes.most_common(1)[0]

        # Check for strong agreement: e.g., 80% of high-confidence predictions agree
        agreement_ratio = count / len(high_conf_predictions)

        return agreement_ratio >= 0.8  # Increased agreement threshold for stability

    def _draw_confidence_overlay(self, image, prediction_text, model_confidence,
                                 pose_confidence, pose_detected, buffer_status=""):
        """Draw confidence information on the image"""
        height, width = image.shape[:2]

        # Background for text - make it slightly larger for buffer status
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Prediction text
        cv2.putText(image, f"Exercise: {prediction_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Model confidence
        conf_color = (0, 255, 0) if model_confidence > 0.7 else (
            0, 255, 255) if model_confidence > 0.5 else (0, 0, 255)
        cv2.putText(image, f"Model Conf: {model_confidence:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

        # Pose detection confidence
        pose_color = (0, 255, 0) if pose_confidence > 0.7 else (
            0, 255, 255) if pose_confidence > 0.5 else (0, 0, 255)
        cv2.putText(image, f"Pose Conf: {pose_confidence:.2f}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)

        # Pose detection status
        status_text = "Pose: OK" if pose_detected else "Pose: WEAK/NONE"  # More descriptive
        status_color = (0, 255, 0) if pose_detected else (0, 0, 255)
        cv2.putText(image, status_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Buffer status
        if buffer_status:
            cv2.putText(image, buffer_status, (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Confidence bars
        self._draw_confidence_bar(
            image, width - 150, 30, model_confidence, "Model")
        self._draw_confidence_bar(
            image, width - 150, 60, pose_confidence, "Pose")

    def _draw_confidence_bar(self, image, x, y, confidence, label):
        """Draw a confidence bar"""
        bar_width = 100
        bar_height = 15

        # Background bar
        cv2.rectangle(image, (x, y), (x + bar_width,
                      y + bar_height), (50, 50, 50), -1)

        # Confidence bar
        conf_width = int(bar_width * confidence)
        color = (0, 255, 0) if confidence > 0.7 else (
            0, 255, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + conf_width,
                      y + bar_height), color, -1)

        # Label
        cv2.putText(image, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run(self, video_path: str = None, confidence_threshold: float = 0.7):
        # Initialize video capture
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"Attempting to open video: {video_path}")
        else:
            cap = cv2.VideoCapture(0)  # Use default webcam if no path provided
            print("Attempting to open webcam.")

        if not cap.isOpened():
            print("Error: Could not open video capture. Exiting.")
            return

        fidx = 0
        # Stores keypoint data (33, 4) or (33, 4) zeros for each frame in the window
        frame_window = []

        # Alternative approach: Use pop method and fixed frame_size for sliding window
        # window_size = 60  # For sliding window approach

        # Current approach: Accumulation with reset
        max_buffer_size = 201  # Maximum accumulation size
        prediction_interval = 30  # Run inference every N frames when buffer has enough data
        min_frames_for_prediction = 60  # Minimum frames needed before starting predictions

        prediction_text = "Uncertain"  # Initial status
        confidence = 0.0
        model_confidence_threshold = 0.6  # Minimum confidence for model predictions
        # Minimum confidence for pose landmarks (MediaPipe's internal visibility score)
        pose_confidence_threshold = 0.5

        recent_predictions = []
        prediction_history_size = 4  # Number of recent predictions to consider for smoothing

        # Optimization: Track buffer quality and inference timing
        frames_since_last_prediction = 0
        last_prediction_confidence = 0.0
        buffer_quality_frames = 0  # Count of non-zero frames in buffer

        # Using a context manager for MediaPipe's PoseLandmarker
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # End of video or error reading frame
                    print("End of video or failed to read frame.")
                    break

                # Convert frame to MediaPipe Image format
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame)

                # Get current timestamp for video processing (more robust than fidx for video)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                pose_landmarker_result = landmarker.detect_for_video(
                    mp_image, timestamp_ms)

                # Extract keypoints and handle confidence
                # self.keypoint_extractor._extract_keypoints is expected to return (33, 4) or None
                current_frame_output = self.keypoint_extractor._extract_keypoints(
                    pose_landmarker_result)

                pose_detected = False
                avg_landmark_confidence = 0.0

                if current_frame_output is not None and current_frame_output.shape == (33, 4):
                    visibility_scores = current_frame_output[:, 3]
                    avg_landmark_confidence = np.mean(visibility_scores)

                    if avg_landmark_confidence >= pose_confidence_threshold:
                        # Append actual keypoints
                        frame_window.append(current_frame_output)
                        buffer_quality_frames += 1
                        pose_detected = True
                    else:
                        # Pose detected but low confidence - append zeros to represent 'no valid pose' for this frame
                        frame_window.append(
                            np.zeros((33, 4), dtype=np.float32))
                else:
                    # No pose detected in the frame at all - append zeros
                    frame_window.append(np.zeros((33, 4), dtype=np.float32))

                # Alternative approach: Maintain sliding window of fixed size using pop method
                # if len(frame_window) > window_size:
                #     removed_frame = frame_window.pop(0)  # Remove oldest frame
                #     # Update quality count if we removed a good frame
                #     if not np.array_equal(removed_frame, np.zeros((33, 4), dtype=np.float32)):
                #         buffer_quality_frames -= 1

                # Current accumulation technique with optimizations
                if len(frame_window) >= max_buffer_size:
                    # If we reach the maximum window size, reset the window but keep some recent frames
                    # Keep the last 50 frames to maintain some temporal context
                    overlap_size = 50
                    frame_window = frame_window[-overlap_size:]

                    # Recalculate buffer quality for the remaining frames
                    buffer_quality_frames = sum(1 for frame in frame_window
                                                if not np.array_equal(frame, np.zeros((33, 4), dtype=np.float32)))

                    print(
                        f"Buffer reset at frame {fidx}. Kept {overlap_size} frames for context.")

                # Optimization: Only run inference at intervals and when we have sufficient data
                should_run_inference = (
                    len(frame_window) >= min_frames_for_prediction and
                    frames_since_last_prediction >= prediction_interval and
                    buffer_quality_frames >= min_frames_for_prediction * 0.6  # At least 60% good frames
                )

                buffer_status = f"Buffer: {len(frame_window)}/{max_buffer_size} (Quality: {buffer_quality_frames})"

                # Run inference when conditions are met
                if should_run_inference:
                    try:
                        # Prepare input tensor and attention mask from the frame_window
                        processed_frames_for_tensor = []
                        mask_values_for_tensor = []

                        for frame_data in frame_window:
                            # If a frame was replaced by zeros (low/no pose), mask it out
                            if np.array_equal(frame_data, np.zeros((33, 4), dtype=np.float32)):
                                processed_frames_for_tensor.append(
                                    # Only XYZ, value of 0
                                    np.zeros((33, 3), dtype=np.float32))
                                mask_values_for_tensor.append(
                                    0.0)  # Mask out this frame
                            else:
                                processed_frames_for_tensor.append(
                                    # Take only XYZ coordinates
                                    frame_data[:, :3])
                                mask_values_for_tensor.append(
                                    1.0)  # This frame is valid

                        # Pad or truncate to exactly 201 frames for model input
                        current_length = len(processed_frames_for_tensor)
                        if current_length < 201:
                            # Pad with zeros
                            padding_needed = 201 - current_length
                            for _ in range(padding_needed):
                                processed_frames_for_tensor.append(
                                    np.zeros((33, 3), dtype=np.float32))
                                mask_values_for_tensor.append(0.0)
                        elif current_length > 201:
                            # Truncate to most recent 201 frames
                            processed_frames_for_tensor = processed_frames_for_tensor[-201:]
                            mask_values_for_tensor = mask_values_for_tensor[-201:]

                        # Convert to PyTorch tensors and add batch dimension (batch_size=1)
                        # Input X: (1, 201, 33, 3)
                        # Mask: (1, 201)
                        X_tensor = torch.tensor(
                            np.array(processed_frames_for_tensor), dtype=torch.float32).unsqueeze(0)
                        Mask_tensor = torch.tensor(
                            np.array(mask_values_for_tensor), dtype=torch.float32).unsqueeze(0)

                        # Move tensors to the correct device (CPU/GPU)
                        X_tensor = X_tensor.to(self.device)
                        Mask_tensor = Mask_tensor.to(self.device)

                        # Run inference with no gradient calculation
                        with torch.no_grad():
                            logits = self.model(
                                X_tensor, temporal_mask=Mask_tensor)

                            probabilities = torch.softmax(logits, dim=1)
                            max_prob, predicted_class_idx = torch.max(
                                probabilities, dim=1)

                            prediction_confidence = max_prob.item()

                            # Set the confidence to the prediction confidence
                            confidence = prediction_confidence
                            last_prediction_confidence = prediction_confidence

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
                                # Stable, just show the class
                                prediction_text = f"{self.class_labels[predicted_class]}"
                            else:
                                # Show the current prediction, but indicate it's uncertain
                                prediction_text = f"{self.class_labels[predicted_class]} (Unstable)"

                            # Print current frame's prediction status
                            print(f"Frame {fidx}: Predicted: {self.class_labels[predicted_class]} "
                                  f"(Model Conf: {prediction_confidence:.3f}, "
                                  f"Pose Conf: {avg_landmark_confidence:.3f}, "
                                  f"Buffer: {len(frame_window)}, Quality: {buffer_quality_frames}, "
                                  f"Stable: {self._is_prediction_stable(recent_predictions, confidence_threshold)})")

                            frames_since_last_prediction = 0  # Reset counter

                    except Exception as e:
                        print(f"Inference error on frame {fidx}: {e}")
                        prediction_text = "Error in prediction"
                        confidence = 0.0  # Reset confidence on error
                        frames_since_last_prediction = 0
                else:
                    # Use last prediction confidence for display
                    confidence = last_prediction_confidence
                    frames_since_last_prediction += 1

                    # Update status based on current state
                    if len(frame_window) < min_frames_for_prediction:
                        # prediction_text = f"Buffering... ({len(frame_window)}/{min_frames_for_prediction})"
                        prediction_text = f"Uncertain"
                    elif buffer_quality_frames < min_frames_for_prediction * 0.6:
                        prediction_text = f"Low Quality ({buffer_quality_frames}/{int(min_frames_for_prediction * 0.6)})"
                    # Otherwise keep the last prediction

                # Draw landmarks on the original frame
                annotated_image = self.draw_landmarks_on_image(
                    frame, pose_landmarker_result)

                # Change prediction text based on model confidence threshold
                if confidence < model_confidence_threshold and prediction_text not in ["Buffering frames...", "Error in prediction"] and not prediction_text.startswith("Buffering") and not prediction_text.startswith("Low Quality"):
                    prediction_text = "Uncertain"

                # Overlay confidence information on the annotated image
                self._draw_confidence_overlay(annotated_image, prediction_text, confidence,
                                              avg_landmark_confidence, pose_detected, buffer_status)

                # Display the frame
                cv2.imshow('Real-Time Exercise Recognition',
                           annotated_image)  # Changed window title

                # Handle key press to exit (wait 1ms for display update)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("'q' pressed. Exiting.")
                    break

                fidx += 1  # Increment frame index

        # Release video capture and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


def launch_gui():
    def select_video():
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            root.destroy()
            real_time_recognizer.run(video_path=file_path)

    def use_camera():
        root.destroy()
        real_time_recognizer.run()  # webcam mode

    model_path = "models/final/hierarchical_transformer_f201_d64_h2_s1_t1_do0.1_20250701_2251.pth"

    global real_time_recognizer
    real_time_recognizer = RealTimeExerciseRecognition(
        model_path=model_path,
        landmarker_model="models/mediapipe/pose_landmarker_full.task"
    )

    # Initialize GUI window
    root = tk.Tk()
    root.title("Exercise Recognition - Select Input")
    root.geometry("300x150")
    root.resizable(False, False)

    # Create buttons
    btn_select_video = tk.Button(
        root, text="Select Video File", width=25, command=select_video)
    btn_select_video.pack(pady=20)

    btn_use_camera = tk.Button(
        root, text="Use Camera", width=25, command=use_camera)
    btn_use_camera.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    # main()
    launch_gui()
