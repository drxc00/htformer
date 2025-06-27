import os
import numpy as np
from datetime import datetime
import torch

class TransformerLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(TransformerLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


def load_model_parameters(filename: str):
    # Take the filename and parse it to get the parameters
    # remove the "hier_transformer_" prefix
    params = filename.replace("hierarchical_transformer_", "")
    params = params.replace(".pth", "")
    
    params = params.split("_")
    
    # Convert to dictionary
    params = {k: int(v) for k, v in zip(params[::2], params[1::2])}
    return params

def generate_model_filename(base_name: str, params: dict, keys_to_include=None, extension='pth'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    if keys_to_include is None:
        # Default: select only key architectural parameters
        keys_to_include = ['num_frames','d_model', 'nhead', 'num_spatial_layers', 'num_temporal_layers', 'dropout']

    # Create short aliases for keys
    key_aliases = {
        'num_frames': 'f',
        'd_model': 'd',
        'nhead': 'h',
        'num_spatial_layers': 's',
        'num_temporal_layers': 't',
        'dropout': 'do',
    }

    parts = [f"{key_aliases[k]}{params[k]}" for k in keys_to_include if k in params]

    filename = f"{base_name}_{'_'.join(parts)}_{timestamp}.{extension}"
    
    # Example output:
    # hier_transformer_f200_d64_h4_s1_t1_do0.1_20250623_1521.pth
    return filename



def process_sample(sample, max_frames: int = 200):
    """
    Processes a sample by padding it to max_frames and creating an attention mask.
    
    Args:
        sample: A numpy array of shape (num_frames, num_keypoints, coordinates)
        max_frames: The maximum sequence length used during training.
    """
    sequence_length = sample.shape[0] 
    X = None
        
    # Pad sequence
    if sequence_length < max_frames:
        pad_len = max_frames - sequence_length
        pad = np.zeros((pad_len, sample.shape[1], sample.shape[2]))
        X = np.concatenate([sample, pad], axis=0)
    else:
        X = sample

    # Create attention mask
    attention_mask = np.zeros(max_frames)
    # Determines which frames are valid
    attention_mask[:sequence_length] = 1
    
    return X, attention_mask, sequence_length


def create_transformer_dataset(data_dir: str = "data/keypoints", 
                                  percentile_cutoff: float = 95.0, verbose: bool = False):
    """
    Create numpy arrays ready for transformer training with proper masking
    
    Returns:
        X: Input sequences [batch_size, max_seq_len, num_keypoints, coordinates]
        y: Labels [batch_size]
        attention_masks: Attention masks [batch_size, max_seq_len]
        sequence_lengths: Original sequence lengths [batch_size]
    """
    labels = {"squats": 0, "deadlifts": 1, "shoulder_press": 2}
    temp_samples = []
    sequence_lengths = []
    
    # Load all samples
    for exercise, label in labels.items():
        folder_path = os.path.join(data_dir, exercise)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue
            
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                path = os.path.join(folder_path, file)
                try:
                    sample = np.load(path)
                    temp_samples.append((sample, label))
                    sequence_lengths.append(sample.shape[0])
                    if verbose:
                        print(f"Loaded {file} from {exercise} with shape {sample.shape}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    
    if not temp_samples:
        raise ValueError("No samples loaded!")
    
    # Determine max_frames using percentile to avoid extreme outliers
    max_frames = int(np.percentile(sequence_lengths, percentile_cutoff))
    print(f"Using max_frames = {max_frames} ({percentile_cutoff}th percentile)")
    print(f"Sequence length stats - Min: {min(sequence_lengths)}, Max: {max(sequence_lengths)}")
    
    # Filter out sequences that are too long
    filtered_samples = []
    filtered_lengths = []
    for i, (sample, label) in enumerate(temp_samples):
        if sequence_lengths[i] <= max_frames:
            filtered_samples.append((sample, label))
            filtered_lengths.append(sequence_lengths[i])
        else:
            print(f"Filtering out sample with length {sequence_lengths[i]}")
    
    print(f"Kept {len(filtered_samples)} out of {len(temp_samples)} samples")
    
    # Create padded arrays with attention masks
    X = []
    y = []
    attention_masks = []
    final_sequence_lengths = []
    
    for sample, label in filtered_samples:
        padded_sample, attention_mask, seq_len = process_sample(sample, max_frames)
        X.append(padded_sample)
        y.append(label)
        attention_masks.append(attention_mask)
        final_sequence_lengths.append(seq_len)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    attention_masks = np.array(attention_masks)
    final_sequence_lengths = np.array(final_sequence_lengths)
    
    print(f"Final dataset shape:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"attention_masks: {attention_masks.shape}")
    print(f"sequence_lengths: {final_sequence_lengths.shape}")
    
    print(f"Kept {len(filtered_samples)} out of {len(temp_samples)} samples")
    num_filtered_out = len(temp_samples) - len(filtered_samples)
    print(f"Number of samples filtered out due to length: {num_filtered_out}") # Add this line
    
    return X, y, attention_masks, final_sequence_lengths