# The KypointExtractor class is used to extract keypoints from video files using MediaPipe.
# This class normalizes the keypoints based on body proportions to ensure scale consistency across different videos.
# 
# The code below defines the KeypointExtractor class, which uses MediaPipe's PoseLandmarker to extract and normalize keypoints from video files.
# It also contains a LegacyKeypointExtractor class which was used for the prototype of the Hierarchical Transformer model. (Do not use this class anymore)

import math
from typing import List
import numpy as np
import cv2 as cv
import mediapipe as mp
from multiprocessing import Process
import os
from scipy.spatial.transform import Rotation as ScipyRotation # For 3D normalization

class KeypointExtractorV2:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.file_count = 0
        self.num_landmarks = 33
        # LANDMARK_INDICES can still be useful for _normalize_pose_3d
        self.LANDMARK_INDICES = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path,
                delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.VIDEO,
            # IMPORTANT: Ensure your model version and options are set to output world landmarks
            # This is usually default for pose_landmarker models, but good to be aware.
        )

    def _normalize_pose_3d(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Normalizes a 3D pose to a canonical orientation and scale.
        Input: landmarks_3d is a (N, 3) NumPy array (MediaPipe BlazePose 33 landmarks).
        Output: Pose with hip_center at origin, person facing +Z, up along +Y, left along +X.
                Spine length (hip_center to shoulder_center) will be normalized to 1.
        """
        if landmarks_3d.shape[0] != self.num_landmarks or landmarks_3d.shape[1] != 3:
            # Handle cases where input is not as expected, though MediaPipe should provide 33,3
            print(f"Warning: Unexpected landmark shape for 3D normalization: {landmarks_3d.shape}")
            return np.zeros((self.num_landmarks, 3))


        # MediaPipe landmark indices from self.LANDMARK_INDICES
        LM_L_SHOULDER, LM_R_SHOULDER = self.LANDMARK_INDICES['left_shoulder'], self.LANDMARK_INDICES['right_shoulder']
        LM_L_HIP, LM_R_HIP = self.LANDMARK_INDICES['left_hip'], self.LANDMARK_INDICES['right_hip']
        LM_NOSE = self.LANDMARK_INDICES['nose']

        # 1. Center the pose at the hip center
        left_hip = landmarks_3d[LM_L_HIP]
        right_hip = landmarks_3d[LM_R_HIP]
        hip_center = (left_hip + right_hip) / 2.0
        centered_landmarks = landmarks_3d - hip_center

        # 2. Define body axes from centered landmarks
        c_l_shoulder = centered_landmarks[LM_L_SHOULDER]
        c_r_shoulder = centered_landmarks[LM_R_SHOULDER]
        c_l_hip = centered_landmarks[LM_L_HIP] # Will be close to -hip_center relative to original hip_center
        c_r_hip = centered_landmarks[LM_R_HIP]
        c_nose = centered_landmarks[LM_NOSE]

        shoulder_midpoint = (c_l_shoulder + c_r_shoulder) / 2.0
        current_hip_midpoint = (c_l_hip + c_r_hip) / 2.0

        body_up_vector = shoulder_midpoint - current_hip_midpoint
        norm_body_up = np.linalg.norm(body_up_vector)
        if norm_body_up < 1e-6: return np.zeros_like(centered_landmarks) # Avoid division by zero for degenerate poses
        body_up_vector /= norm_body_up

        body_left_vector = c_l_shoulder - c_r_shoulder # From right to left is person's left
        norm_body_left = np.linalg.norm(body_left_vector)
        if norm_body_left < 1e-6: return np.zeros_like(centered_landmarks)
        body_left_vector /= norm_body_left

        body_left_vector -= np.dot(body_left_vector, body_up_vector) * body_up_vector # Make orthogonal to up
        norm_body_left_ortho = np.linalg.norm(body_left_vector)
        if norm_body_left_ortho < 1e-6: return np.zeros_like(centered_landmarks)
        body_left_vector /= norm_body_left_ortho

        body_forward_vector = np.cross(body_up_vector, body_left_vector) # up x left = forward
        # Disambiguate Forward Vector
        chest_to_nose_vector = c_nose - shoulder_midpoint
        if np.linalg.norm(chest_to_nose_vector) > 1e-6 and np.dot(body_forward_vector, chest_to_nose_vector) < 0:
            body_forward_vector = -body_forward_vector
            body_left_vector = -body_left_vector

        # 3. Align to canonical axes (Up=+Y, Forward=+Z, Left=+X)
        source_vectors = np.array([body_up_vector, body_forward_vector])
        target_vectors = np.array([[0, 1, 0], [0, 0, 1]]) # Align body_up with Y, body_forward with Z
        
        try:
            rotation, _ = ScipyRotation.align_vectors(target_vectors, source_vectors)
            oriented_landmarks = rotation.apply(centered_landmarks)
        except Exception as e:
            # print(f"Warning: Could not align vectors: {e}. Returning centered landmarks.")
            oriented_landmarks = centered_landmarks


        # 4. Scale normalization (spine length = 1)
        o_l_shoulder = oriented_landmarks[LM_L_SHOULDER]
        o_r_shoulder = oriented_landmarks[LM_R_SHOULDER]
        o_l_hip = oriented_landmarks[LM_L_HIP]
        o_r_hip = oriented_landmarks[LM_R_HIP]
        
        o_shoulder_midpoint = (o_l_shoulder + o_r_shoulder) / 2.0
        o_hip_midpoint = (o_l_hip + o_r_hip) / 2.0

        spine_length = np.linalg.norm(o_shoulder_midpoint - o_hip_midpoint)
        
        if spine_length > 1e-6:
            normalized_scaled_landmarks = oriented_landmarks / spine_length
        else:
            normalized_scaled_landmarks = oriented_landmarks

        return normalized_scaled_landmarks

    def _load_video(self, path: str) -> cv.VideoCapture:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video does not exist: {path}")
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        return cap
    
    def _extract_video_data(self, cap: cv.VideoCapture):
        orig_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        return orig_w, orig_h, total_frames

    def _extract_keypoints(self, detection_result):
        current_frame_output = np.zeros((self.num_landmarks, 4))
        if detection_result.pose_world_landmarks and detection_result.pose_landmarks:
            if len(detection_result.pose_world_landmarks) > 0 and \
                len(detection_result.pose_landmarks[0]) == self.num_landmarks and \
                len(detection_result.pose_world_landmarks[0]) == self.num_landmarks:
                
                world_landmarks_mp = detection_result.pose_world_landmarks[0]
                image_landmarks_mp = detection_result.pose_landmarks[0] # For visibility

                # Convert world landmarks to (33, 3) NumPy array
                frame_world_keypoints_3d = np.array(
                    [[lm.x, lm.y, lm.z] for lm in world_landmarks_mp]
                )
                
                # Apply 3D normalization
                normalized_3d_keypoints = self._normalize_pose_3d(frame_world_keypoints_3d)
                
                # Get visibility scores
                visibilities = np.array(
                    [[lm.visibility] for lm in image_landmarks_mp] # Shape (33,1)
                )
                
                # Combine normalized 3D keypoints with visibility
                current_frame_output = np.hstack((normalized_3d_keypoints, visibilities))
                
        return current_frame_output

    def extract(self, file: str) -> np.ndarray:
        try:
            with self.PoseLandmarker.create_from_options(self.options) as landmarker:
                cap = self._load_video(file)
                
                # Extract video properties
                orig_w, orig_h, total_frames = self._extract_video_data(cap)
                
                all_frames_keypoints = []
                frame_idx = 0
                
                print(f"Processing {file}: {orig_w}x{orig_h}, {total_frames} frames")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    # Pass frame_idx as timestamp for video mode
                    timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC)) 
                    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms) # Use timestamp

                    # Extract keypoints from the detection result
                    current_frame_output = self._extract_keypoints(detection_result)
                    
                    all_frames_keypoints.append(current_frame_output)
                    frame_idx += 1
                
                cap.release()
                print(f"Extracted and normalized {len(all_frames_keypoints)} frames from {file}")
                return np.array(all_frames_keypoints)
        except Exception as e:
            print(f"✗ Error processing {file}: {str(e)}")
            # Return a consistent empty or error structure if needed, e.g., empty array
            return np.empty((0, self.num_landmarks, 4))

class KeypointExtractor:
    def __init__ (self, model_path: str):
        self.model_path = model_path
        
        # Filename Counter
        self.file_count = 0
        self.num_landmarks = 33
        
        # MediaPipe landmark indices
        # These indices are used for normalization and consistency
        self.LANDMARK_INDICES = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }

        # Model config
        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path,
                delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.VIDEO
        )
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_normalization_factor(self, landmarks: List[List[float]]) -> float:
        """
        Calculate normalization factor based on body proportions
        Returns a scale factor to normalize pose size
        """
        left_shoulder = landmarks[self.LANDMARK_INDICES['left_shoulder']]
        right_shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder']]
        left_hip = landmarks[self.LANDMARK_INDICES['left_hip']]
        right_hip = landmarks[self.LANDMARK_INDICES['right_hip']]
        
        shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, 
                            (left_shoulder[1] + right_shoulder[1])/2]
        hip_center = [(left_hip[0] + right_hip[0])/2, 
                        (left_hip[1] + right_hip[1])/2]
        
        torso_length = self._calculate_distance(shoulder_center, hip_center)
        return torso_length if torso_length > 0 else 1.0
    
    def _normalize_pose_scale(self, landmarks: List[List[float]]) -> List[List[float]]:
        """
        Normalize pose scale based on body proportions
        """
        normalization_factor = self._get_normalization_factor(landmarks)
        
        if normalization_factor <= 0:
            return landmarks
        
        # Calculate pose center (average of all visible landmarks)
        visible_landmarks = [lm for lm in landmarks if lm[3] > 0.5]  # visibility > 0.5
        if not visible_landmarks:
            return landmarks
            
        center_x = sum(lm[0] for lm in visible_landmarks) / len(visible_landmarks)
        center_y = sum(lm[1] for lm in visible_landmarks) / len(visible_landmarks)
        
        # Normalize landmarks relative to center and scale
        normalized_landmarks = []
        target_scale = 0.3  # Target normalized scale (adjustable)
        scale_factor = target_scale / normalization_factor
        
        for landmark in landmarks:
            if landmark[3] > 0:  # If landmark is visible
                # Translate to origin, scale, then translate back
                norm_x = center_x + (landmark[0] - center_x) * scale_factor
                norm_y = center_y + (landmark[1] - center_y) * scale_factor
                normalized_landmarks.append([norm_x, norm_y, landmark[2], landmark[3]])
            else:
                normalized_landmarks.append(landmark)  # Keep invisible landmarks as-is
        
        return normalized_landmarks
    
    def _load_video(self, path: str) -> cv.VideoCapture:
        """Load video file and return VideoCapture object"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video does not exist: {path}")
        
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        
        return cap
    
    def extract(self, file: str) -> np.ndarray:
        """
        Extract scale-consistent keypoints from video
        
        Returns:
            dict: Processing results and statistics
        """
        try:
            with self.PoseLandmarker.create_from_options(self.options) as landmarker:
                cap = self._load_video(file)
                
                # Get video properties
                orig_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                
                keypoints = []
                frame_idx = 0
                
                print(f"Processing {file}: {orig_w}x{orig_h}, {total_frames} frames")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process at original resolution (no resizing for better accuracy)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    detection_result = landmarker.detect_for_video(mp_image, frame_idx)
                    
                    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                        landmarks = detection_result.pose_landmarks[0]
                        
                        # Convert to list format
                        frame_keypoints = [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility] 
                            for landmark in landmarks
                        ]
                        
                        # Apply scale normalization
                        normalized_keypoints = self._normalize_pose_scale(frame_keypoints)
                        keypoints.append(normalized_keypoints)
                    else:
                        # No landmarks detected
                        frame_keypoints = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.num_landmarks)]
                        keypoints.append(frame_keypoints)
                    
                    frame_idx += 1
                
                cap.release()
                
                print(f"Extracted {len(keypoints)} frames from {file}")
                
                # Convert to numpy array
                keypoints_array = np.array(keypoints)
            
                return keypoints_array 
        except Exception as e:
            print(f"✗ Error processing {file}: {str(e)}")
            return {'success': False, 'filename': file, 'error': str(e)}

class LegacyKeypointExtractor:
    def __init__(self, model_path: str , input_dir: str, output_dir: str, exercise: str):
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.exercise = exercise # exercise type
        
        # Filename Counter
        self.file_count = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Model config
        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode:
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO)
        
    def _load_video(self, file_name: str) -> cv.VideoCapture:
        p = os.path.join(self.input_dir, file_name) # get full path of file
        if not os.path.exists(p):
            raise ValueError("Video does not exist")
        return cv.VideoCapture(p)
    
    def _resize(self, frame, target_size=(256, 256)):
        h, w = frame.shape[:2]
        scale = min(target_size[0]/h, target_size[1]/w)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv.resize(frame, (new_w, new_h))
        
        # Create a blank canvas and paste the resized frame
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas
    
    def extract(self, file_name:str) -> None:
        """
        Extract keypoints from video
        This assumes that the filename is under the input_dir
        """
        
        landmarker = self.PoseLandmarker.create_from_options(self.options)
        cap = self._load_video(file_name)
        keypoints = []
        frame_idx = 0
        while cap.isOpened():
            r, frame = cap.read()
            if not r: break
            
            rf = self._resize(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rf)
            detection_result = landmarker.detect_for_video(mp_image, frame_idx)
            pose_landmarks_list = detection_result.pose_landmarks
            
            # Convert landmarks to a numpy-friendly format
            frame_keypoints = []
            if pose_landmarks_list:
                for landmark in pose_landmarks_list[0]:
                    frame_keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                # If no landmarks detected, add a placeholder (all zeros)
                frame_keypoints = np.zeros((33, 4))  # MediaPipe pose has 33 landmarks with x,y,z,visibility
                
            keypoints.append(frame_keypoints)
            frame_idx+=1

        # Save keypoints to file
        output_path = os.path.join(self.output_dir, self.exercise + "_" + str(self.file_count) + ".npy")
        np.save(output_path, np.array(keypoints))
        
        self.file_count += 1
        
        cap.release()

        print("Keypoints extracted from " + file_name)
        
    def run(self) -> None:
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith(".mp4"):
                self.extract(file_name)
   
   
def extract_keypoints(input_dir, output_dir, exercise):
    """
    Extract keypoints from video files in the input directory and save them to the output directory.
    
    Args:
        input_dir (str): Path to the input directory containing video files.
        output_dir (str): Path to the output directory to save the extracted keypoints.
        exercise (str): Name of the exercise to extract keypoints for.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create KeypointExtractor instance
    extractor = KeypointExtractorV2(model_path="models/mediapipe/pose_landmarker_full.task")
    
    file_count = 0  # Reset file count for each extraction session
    
    # Loop through all files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            # Extract keypoints from the file
            keypoints = extractor.extract(os.path.join(input_dir, file))
            
            # Save keypoints to a NumPy file
            output_path = os.path.join(output_dir, exercise + "_" + str(file_count) + ".npy")
            np.save(output_path, keypoints)
            
            file_count += 1
            
            print(f"Extracted keypoints for {file}")
            
def extract_deadlift():
    input_dir = "data/augmented/deadlifts/"
    output_dir = "data/keypoints/deadlifts/"
    exercise = "deadlift"
    
    extract_keypoints(input_dir, output_dir, exercise)
    
def extract_squat():
    input_dir = "data/augmented/squats/"
    output_dir = "data/keypoints/squats/"
    exercise = "squat"
    
    extract_keypoints(input_dir, output_dir, exercise)         

def main () -> None:
    p1 = Process(target=extract_deadlift)
    p2 = Process(target=extract_squat)
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
        
if __name__ == '__main__':
    main()