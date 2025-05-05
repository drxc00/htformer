import numpy as np
import cv2 as cv
import mediapipe as mp
from multiprocessing import Process
import time
import os

class KeypointExtractor:
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
            
def extract_deadlifts() -> None:
    extractor = KeypointExtractor(
        input_dir="data\\augmented\\deadlifts",
        output_dir="data\\keypoints\\deadlifts",
        model_path="models\\mediapipe\\pose_landmarker_heavy.task",
        exercise="deadlift",
    )
    extractor.run()
    
def extract_squats() -> None:
    extractor = KeypointExtractor(
        input_dir="data\\augmented\\squats",
        output_dir="data\\keypoints\\squats",
        model_path="models\\mediapipe\\pose_landmarker_heavy.task",  
        exercise="squat",
    )
    extractor.run()

def main () -> None:
    start_time = time.time()
    print("Extracting started")
    deadlift_proc = Process(target=extract_deadlifts)
    squat_proc = Process(target=extract_squats)

    deadlift_proc.start()
    squat_proc.start()
    deadlift_proc.join()
    squat_proc.join()
    
    end_time = time.time() # get en
    
    print("Extracting finished in " + str(end_time - start_time) + " seconds")
            
        
        
if __name__ == '__main__':
    main()