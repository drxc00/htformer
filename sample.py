
from mediapipe import solutions
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 
import os

FOLDER_DIR =  "data/augmented/squats/"
MODEL_PATH =  'models/mediapipe/pose_landmarker_heavy.task'
OUTPUT_PATH = 'output/sample/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    
    

def load_video(v_name: str) -> cv2.VideoCapture:
    """ Loads a video from a folder and returns a cv2.VideoCapture object. """
    p: str = FOLDER_DIR + v_name
    # Check if the video exists
    if not os.path.exists(p):
        raise ValueError("Video does not exist")
    
    return cv2.VideoCapture(p)

def resize(image, target_size=(256, 256)):
    h, w = image.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create a blank canvas and paste the resized image
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def draw_landmarks_on_image(rgb_image, detection_result) -> np.ndarray:
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    # Loop through the landmarks and draw them on the image
    for idx in range (len(pose_landmarks_list)):
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


def main() -> None:
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO)
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        # load video file from folder
        file_name = "32903_8_original.mp4"
        cap = load_video(file_name)
        out = cv2.VideoWriter(OUTPUT_PATH + file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (256, 256))
        
        f = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            # perform inference
            resized_frame = resize(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_frame)
            detection_result = landmarker.detect_for_video(mp_image, f)

            # draw landmarks on image
            annotated_image = draw_landmarks_on_image(resized_frame, detection_result)

            # write annotated image to video
            out.write(annotated_image)
            
            f += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        

if __name__ == '__main__':
    main()
    