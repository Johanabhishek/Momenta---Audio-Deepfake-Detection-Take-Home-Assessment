# RawNet2 Architecture Diagrams and Implementation Flow

This document provides visual representations and additional pseudocode to complement the RawNet2 implementation approach for audio deepfake detection.

## RawNet2 Architecture Diagram

```
                                 RawNet2 Architecture
                                 
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Raw Audio    │     │  Sinc Filter  │     │  Residual     │     │  Attention    │
│  Waveform     │────▶│  Layer        │────▶│  Blocks       │────▶│  Mechanism    │
│  (64000 samp) │     │  (128 filters)│     │  (Multiple)   │     │  (Freq. focus)│
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                                                                          │
                                                                          ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Output       │     │  Fully        │     │  GRU Layer    │     │  Feature      │
│  (Genuine/    │◀────│  Connected    │◀────│  (Temporal    │◀────│  Normalization│
│   Spoofed)    │     │  Layers       │     │   modeling)   │     │  (BatchNorm)  │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

## Detailed Component Diagrams

### 1. Sinc Filter Layer

```
                         Sinc Filter Layer
                         
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Raw Audio    │     │  Parameterized│     │  Filtered     │
│  Waveform     │────▶│  Band-pass    │────▶│  Signal       │
│               │     │  Filters      │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
                             │
                      ┌──────┴──────┐
                      │ Learnable   │
                      │ Parameters: │
                      │ - Low cutoff│
                      │ - Band width│
                      └─────────────┘
```

### 2. Residual Block

```
                         Residual Block
                         
                      ┌───────────────┐
                      │  Input        │
                      │  Features     │
                      └───────┬───────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
┌───────────────────────────┐    ┌──────────────────┐
│  Main Path                │    │ Shortcut Path    │
│  ┌─────────────────────┐  │    │                  │
│  │ Conv1D + BatchNorm  │  │    │ Conv1D (1x1)     │
│  │ + LeakyReLU         │  │    │                  │
│  └──────────┬──────────┘  │    │                  │
│             │             │    │                  │
│  ┌──────────▼──────────┐  │    │                  │
│  │ Conv1D              │  │    │                  │
│  └──────────┬──────────┘  │    │                  │
│             │             │    │                  │
│  ┌──────────▼──────────┐  │    │ ┌──────────────┐ │
│  │ MaxPool1D           │  │    │ │ MaxPool1D    │ │
│  └──────────┬──────────┘  │    │ └──────┬───────┘ │
└─────────────┬─────────────┘    └────────┬─────────┘
              │                           │
              └────────────┬─────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Output Features  │
                  │ (Addition)       │
                  └──────────────────┘
```

### 3. Attention Mechanism

```
                      Attention Mechanism
                      
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Feature Maps │     │  Global Avg   │     │  FC Layer     │
│  from Residual│────▶│  Pooling      │────▶│               │
│  Blocks       │     │               │     │               │
└───────┬───────┘     └───────────────┘     └───────┬───────┘
        │                                           │
        │                                           │
        │                                   ┌───────▼───────┐
        │                                   │  Sigmoid      │
        │                                   │  Activation   │
        │                                   └───────┬───────┘
        │                                           │
        │                                           │
┌───────▼───────────────────────────────────────────▼───────┐
│                      Element-wise                          │
│                      Multiplication                        │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Attention-weighted│
                  │ Feature Maps     │
                  └──────────────────┘
```

### 4. GRU Layer

```
                           GRU Layer
                           
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Feature Maps │     │  Reshape      │     │  GRU Cells    │
│  (batch,      │────▶│  (batch,      │────▶│  (3 layers)   │
│   filt, time) │     │   time, filt) │     │               │
└───────────────┘     └───────────────┘     └───────┬───────┘
                                                    │
                                                    │
                                            ┌───────▼───────┐
                                            │  Last Hidden  │
                                            │  State        │
                                            └───────┬───────┘
                                                    │
                                                    │
                                            ┌───────▼───────┐
                                            │  FC Layers    │
                                            │  for          │
                                            │  Classification│
                                            └───────────────┘
```

## Implementation Flow Diagram

```
                      Implementation Flow
                      
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Data         │     │  Model        │     │  Training     │
│  Preparation  │────▶│  Initialization│────▶│  Loop        │
│               │     │               │     │               │
└───────────────┘     └───────────────┘     └───────┬───────┘
                                                    │
                                                    │
                                            ┌───────▼───────┐
                                            │  Evaluation   │
                                            │  (EER, t-DCF) │
                                            └───────┬───────┘
                                                    │
                                                    │
                                            ┌───────▼───────┐
                                            │  Deployment   │
                                            │  & Inference  │
                                            └───────────────┘
```

## Additional Pseudocode

### Data Preprocessing Function

```python
def preprocess_audio(audio_path, target_sr=16000, target_length=64000):
    """
    Preprocess audio file for RawNet2 model.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate
        target_length: Target number of samples
        
    Returns:
        Preprocessed audio as numpy array
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Handle length
    if len(audio) < target_length:
        # Pad if too short
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        # Center crop if too long
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    
    return audio
```

### Batch Inference Function

```python
def batch_inference(model, audio_files, batch_size=32):
    """
    Perform batch inference on multiple audio files.
    
    Args:
        model: Trained RawNet2 model
        audio_files: List of audio file paths
        batch_size: Batch size for inference
        
    Returns:
        Array of scores (probability of being spoofed)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Preprocess audio files
    preprocessed_audio = []
    for audio_path in audio_files:
        audio = preprocess_audio(audio_path)
        preprocessed_audio.append(audio)
    
    # Create batches
    num_batches = (len(preprocessed_audio) + batch_size - 1) // batch_size
    all_scores = []
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(preprocessed_audio))
            
            # Create batch tensor
            batch_audio = preprocessed_audio[start_idx:end_idx]
            batch_tensor = torch.FloatTensor(batch_audio).unsqueeze(1).to(device)  # Add channel dimension
            
            # Forward pass
            outputs = model(batch_tensor, is_test=True)
            
            # Get scores
            scores = outputs[:, 1].cpu().numpy()  # Probability of being spoofed
            all_scores.append(scores)
    
    return np.concatenate(all_scores)
```

### Real-time Processing Function

```python
def process_audio_stream(model, stream_source, window_size=64000, hop_size=16000):
    """
    Process audio stream in real-time for deepfake detection.
    
    Args:
        model: Trained RawNet2 model
        stream_source: Audio stream source
        window_size: Window size in samples
        hop_size: Hop size in samples
        
    Returns:
        Generator yielding (timestamp, score) tuples
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize buffer
    buffer = np.zeros(window_size)
    
    # Process stream
    timestamp = 0
    with torch.no_grad():
        for chunk in stream_source:
            # Update buffer
            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk
            
            # Process if buffer is full
            if timestamp * hop_size >= window_size:
                # Preprocess
                audio = buffer / (np.max(np.abs(buffer)) + 1e-8)
                
                # Convert to tensor
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward pass
                output = model(audio_tensor, is_test=True)
                
                # Get score
                score = output[0, 1].cpu().numpy()  # Probability of being spoofed
                
                # Yield result
                yield (timestamp * hop_size / 16000, score)  # Convert to seconds
            
            # Update timestamp
            timestamp += 1
```

## Model Performance Visualization

```python
def visualize_model_performance(genuine_scores, spoofed_scores):
    """
    Visualize model performance with score distributions and ROC curve.
    
    Args:
        genuine_scores: Scores for genuine audio samples
        spoofed_scores: Scores for spoofed audio samples
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot score distributions
    ax1.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine')
    ax1.hist(spoofed_scores, bins=50, alpha=0.5, label='Spoofed')
    ax1.set_title('Score Distributions')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # Prepare data for ROC curve
    y_true = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(spoofed_scores))])
    y_score = np.concatenate([genuine_scores, spoofed_scores])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Find EER
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    # Plot ROC curve
    ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f}, EER = {eer:.3f})')
    ax2.plot([0, 1], [1, 0], 'k--')  # Random guess line
    ax2.scatter(eer, 1-eer, color='red', label=f'EER = {eer:.3f}')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()
```

These diagrams and additional pseudocode provide a more comprehensive understanding of the RawNet2 architecture and implementation approach for audio deepfake detection.
