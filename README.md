# Hierarchical Transformer for Exercise Recognition

This repository contains the implementation of a hierarchical transformer model for recognizing weightlifting exercises (deadlifts, squats, shoulder press) from pose keypoint data.

## Project Structure

### Core Modules
- `core/`
  - `augment.py`: Data augmentation (flips, rotations) for exercise videos
  - `keypoint_extractor.py`: Extracts body keypoints using MediaPipe
  - `utils.py`: Utility functions for data processing and model operations
  - `models/`
    - `base_transformer_model.py`: Base transformer model implementation
    - `hierarchical_transformer.py`: Hierarchical transformer implementation

### Notebooks
- `notebooks/`
  - `training/`
    - `hierarchical_transformer_training.ipynb`: Main training notebook
    - `hierarchical_transformer_prototype.ipynb`: Prototype implementation
  - `others/`
    - `test_trained_model.ipynb`: Model evaluation notebook
    - `visualization.ipynb`: Data visualization tools
  - `create_dataset.ipynb`: Dataset creation pipeline
  - `extract_keypoints.ipynb`: Keypoint extraction process

### Data Organization
- `data/`
  - `raw/`: Original exercise videos
  - `raw_uncut/`: Unprocessed full-length videos
  - `keypoints/`: Extracted pose keypoints
  - `augmented/`: Augmented video frames
  - `unseen/`: Test data not used in training

### Models
- `models/hierarchical_transformer/`: Contains trained model weights
  - Multiple versions with training dates
  - Latest: `models/hierarchical_transformer/hierarchical_transformer_f200_d64_h4_s1_t1_do0.1_20250623_1841.pth`

## Requirements
- Python 3.8+
- PyTorch
- MediaPipe
- NumPy
- Jupyter Notebook

## Usage

1. Data Preparation: First prepare the data following the structure described in the `Data Organization` section. Keypoint extraction can be done using the `extract_keypoints.ipynb` notebook. This will perform augmentation and pose detection and extract keypoints for all exercise videos in the `data/raw` directory. The output will be saved in `data/keypoints/{exercise_name}/`.

2. Training: Run the `hierarchical_transformer_training.ipynb` notebook. This will train the model on the prepared data and save the trained weights in `models/hierarchical_transformer/`.
