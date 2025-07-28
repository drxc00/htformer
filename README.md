# Hierarchical Transformer for Exercise Recognition

This repository contains the implementation of a hierarchical transformer model for recognizing weightlifting exercises (deadlifts, squats, shoulder press) from pose keypoint data extracted using MediaPipe.

## Model Architecture

The hierarchical transformer architecture consists of:
- **Spatial Encoder**: Processes individual frames to capture body pose relationships
- **Temporal Encoder**: Models exercise movement patterns across time
- **Classification Head**: Outputs exercise predictions with confidence scores

Key hyperparameters:
- Embedding dimension: 64/128
- Number of heads: 2/4
- Dropout rate: 0.1-0.4
- Sequence length: 200 frames

## Project Structure

### Core Modules
- `core/`
  - `augment.py`: Data augmentation (flips, rotations) for exercise videos
  - `keypoint_extractor.py`: Extracts body keypoints using MediaPipe (33 keypoints per frame)
  - `utils.py`: Utility functions for data processing and model operations
  - `models/`
    - `base_transformer_model.py`: Base transformer model implementation
    - `hierarchical_transformer.py`: Hierarchical transformer implementation
    - `hierarchical_transformer_prototype.py`: Prototype implementation

### Notebooks
- `notebooks/`
  - `training/`
    - `hierarchical_transformer_training.ipynb`: Main training notebook (includes learning rate scheduling, early stopping)
    - `hierarchical_transformer_prototype.ipynb`: Prototype implementation
    - `base_transformer_model.ipynb`: Base model training
    - `kfold_test.ipynb`: K-fold cross validation testing (5-fold)
  - `others/`
    - `test_trained_model.ipynb`: Model evaluation notebook (precision/recall metrics)
    - `visualization.ipynb`: Data visualization tools (keypoint plotting)
    - `mediapipe_analysis.ipynb`: MediaPipe analysis (confidence scores)
    - `model_parameters.ipynb`: Model parameter analysis
  - `create_dataset.ipynb`: Dataset creation pipeline
  - `extract_keypoints.ipynb`: Keypoint extraction process
  - `test_real_world_inference.ipynb`: Real-world inference testing

### Data Organization
- `data/`
  - `raw/`: Original exercise videos (MP4 format, 30fps)
  - `raw_uncut/`: Unprocessed full-length videos
  - `keypoints/`: Extracted pose keypoints (JSON format)
  - `augmented/`: Augmented video frames
  - `unseen/`: Test data not used in training

### Models
- `models/`
  - `base_hierarchical_transformer/`: Base model weights
  - `final/`: Final trained model weights (best performing)
  - `hierarchical_transformer/`: Various trained hierarchical transformer versions
  - `hierarchical transformer/`: Legacy model weights
  - `mediapipe/`: MediaPipe model files

## Performance Metrics

Best model achieves on validation set (unseen data):
- Accuracy: 92.4%
- Precision: 93.1% 
- Recall: 91.8%
- F1-score: 92.4%

Best model achieves on test set:
- Accuracy: 99.0%
- Precision: 99.0%
- Recall: 99.0%
- F1-score: 99.0%

## Requirements
- Python 3.8+
- PyTorch 2.0+
- MediaPipe 0.10+
- NumPy 1.23+
- OpenCV (for video processing)
- Matplotlib (for visualization)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place exercise videos in `data/raw/{exercise_name}/` (supported formats: MP4, MOV)
2. Run `notebooks/extract_keypoints.ipynb` to:
   - Extract pose keypoints using MediaPipe
   - Perform data augmentation
   - Save processed data to `data/keypoints/`

### Training
1. Configure training parameters in `notebooks/training/hierarchical_transformer_training.ipynb`
2. Run all cells to:
   - Load and preprocess data
   - Train model with early stopping
   - Save best weights to `models/hierarchical_transformer/`

### Inference Options

#### Real-time Demo
```bash
python real_time_demo.py --model_path models/final/hierarchical_transformer_f201_d64_h2_s1_t1_do0.1_20250701_1555.pth
```

#### Video File Inference
1. Use `infer_from_video.ipynb` to:
   - Process video files
   - Display predictions with confidence scores
   - Save annotated output videos

#### Testing
- `notebooks/others/test_trained_model.ipynb`: Quantitative evaluation
- `notebooks/test_real_world_inference.ipynb`: Qualitative testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
