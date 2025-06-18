# This file contains the DataAugmentor class, which is used to augment video files by applying horizontal flips, slight rotations, and resizing.
# To use the Augmentor, create an instance of the DataAugmentor class with the input and output directories, and call the `run` method.
# More augmentation methods can be added as needed. Currently, it supports only defined in the thesis proposal: horizontal flip and slight rotation.

import time
from tracemalloc import start
import numpy as np
import os
import cv2 as cv
import threading

class DataAugmentor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    
    def _load_video(self, file_name: str) -> cv.VideoCapture:
        p = os.path.join(self.input_dir, file_name) # get full path of file
        
        if not os.path.exists(p):
            raise ValueError("Video does not exist")
        
        return cv.VideoCapture(p)
    
    def _apply_horizontal_flip(self, frame: np.ndarray) -> np.ndarray:
        return cv.flip(frame, 1)
    
    def _create_video_writer(self, file_name: str, w: int, h: int) -> cv.VideoWriter:
        output_path = os.path.join(self.output_dir, file_name)
        return cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    def augment(self, file_name: str) -> None:
        cap = self._load_video(file_name)
        
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        base_name = os.path.splitext(file_name)[0] # remove extension from file nam
        
        writers = {
            "original": self._create_video_writer(f"{base_name}_original.mp4", w=width, h=height),
            "flip": self._create_video_writer(f"{base_name}_aug_hflip.mp4", w=width, h=height),
        }
        
        while True:
            r, f = cap.read()
            
            if not r: break
            
            # Save original frame
            writers["original"].write(f)
            
            img_flip = self._apply_horizontal_flip(f)

            writers["flip"].write(img_flip)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        
        for w in writers.values():
            w.release()
        
        cv.destroyAllWindows()
        
        print(f"[INFO] Done. File: {file_name} augmented.")
    
    def run(self) -> None:
        """ Runs the augmentor on all videos in the input directory. """
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith(".mp4"):
                self.augment(file_name)
                

def augment_squats():
    augmentor = DataAugmentor('data\\raw\\squats', 'data\\augmented\\squats')
    augmentor.run()
    
def augment_deadlifts():
    augmentor = DataAugmentor('data\\raw\\deadlifts', 'data\\augmented\\deadlifts')
    augmentor.run()
    
def augment_shoulder_press():
    augmentor = DataAugmentor('data\\raw\\shoulder_press', 'data\\augmented\\shoulder_press')
    augmentor.run()
            
def main() -> None:
    start = time.time()
    print ("[INFO] Starting augmentation...")
    t_1 = threading.Thread(target=augment_squats)
    t_2 = threading.Thread(target=augment_deadlifts)
    t_3 = threading.Thread(target=augment_shoulder_press)
    
    t_1.start()
    t_2.start()
    t_3.start()
    
    t_1.join()
    t_2.join()
    t_3.join()
    
    end = time.time()
    
    print ("[INFO] Done. Time taken: ", end - start, "s")

if __name__ == '__main__':
    main()
        