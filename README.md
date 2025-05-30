# Please ayoko na mag thesis
This repository contains the files and code used for creating the hierarchical transformer model for exercise recognition.

## Data Organization

Google drive link: [Data](https://drive.google.com/drive/folders/1clWxd97NM0EExJJRppgwfDhhTDBVFVX8?usp=sharing)

- `data/raw/`: Contains the original unprocessed images
- `data/keypoints/`: Stores extracted keypoint data. The `deadlifts_squat.npz` file contains the keypoint with label data for the deadlifts and squat exercises.
- `data/augmented/`: Contains augmented images generated from the original dataset
- `data/raw_unseen/`: Contains raw videos that were not used in the dataset creation process

## File Structure
- `augment.py`: Script for data augmentation that creates variations of exercise videos (flips, rotations)
- `keypoint_extractor.py`: Extracts body keypoints from augmented videos using MediaPipe
- `create_dataset.ipynb`: Jupyter notebook that processes extracted keypoints into a complete dataset
- `transformer_prototype.ipynb`: Main implementation of the hierarchical transformer model for exercise recognition
- `test_spatial.ipynb`: Testing notebook for the spatial transformer component
- `test.py`: Simple test script for the augmentation process
- `testing_again.py`: Script for testing MediaPipe pose detection visualization
- `models/`: Directory containing model files
  - `mediapipe/`: Contains MediaPipe pose detection models
- `data/`: Directory containing all data files
  - `raw/`: Original exercise videos
  - `augmented/`: Augmented videos generated from raw data
  - `keypoints/`: Extracted keypoint data from augmented videos

## How to use
1. If you have time, you can start with the raw data lol. Use the `augment.py` file to augment the data.
2. After augmenting, you will notice a new folder called `augmented` in the `data` folder.
3. Use the `keypoint_extractor.py` file to extract keypoint data from the augmented images.
4. After extracting, you will notice a new folder called `keypoints` in the `data` folder.
5. To create the complete dataset, run the `create_dataset.ipynb` file.
6. Then go crazy with model.