# RawNet2 Model Architecture Components

This document provides a detailed breakdown of the key components in the RawNet2 architecture for audio deepfake detection, based on the original implementation.

## Configuration Parameters

From the model configuration file (`model_config_RawNet2.yaml`), we can extract the following key parameters:

```yaml
model:
  margin: 2
  nb_samp: 64000          # Number of samples in input audio
  first_conv: 128         # Number of filter coefficients in first conv layer
  in_channels: 1          # Input channels (mono audio)
  filts: [128, [128, 128], [128, 512], [512, 512]]  # Filter configurations for residual blocks
  blocks: [2, 4]          # Number of blocks in each layer
  nb_fc_node: 1024        # Number of nodes in fully connected layer
  gru_node: 1024          # Number of GRU units
  nb_gru_layer: 3         # Number of GRU layers
  nb_classes: 2           # Number of output classes (genuine/spoofed)
```

## Component 1: Sinc Filter Layer

The sinc filter layer is the first layer of the network and is responsible for processing the raw audio waveform.

### Implementation Details:

```python
class SincConv(nn.Module):
    def __init__(self, device, out_channels, kernel_size, in_channels=1):
        super(SincConv, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        
        # Initialize filter bands (low and high cutoff frequencies)
        self.low_hz = nn.Parameter(torch.Tensor(out_channels))
        self.band_hz = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize with mel-scale filter bank
        self.low_hz.data.uniform_(0, 0.5)  # Normalized frequencies [0, 0.5]
        self.band_hz.data.uniform_(0, 0.5)
        
        # Window for sinc function (Hamming window)
        self.window = torch.hamming_window(kernel_size)
        
    def forward(self, x):
        # Convert parameters to actual filter frequencies
        low = self.min_low_hz + torch.abs(self.low_hz) * (self.min_low_hz - self.min_high_hz)
        high = low + self.min_band_hz + torch.abs(self.band_hz) * (self.max_band_hz - self.min_band_hz)
        
        # Create filter bank
        band = (high - low)[:, None]
        f_times_t = torch.linspace(0, kernel_size - 1, steps=kernel_size, device=self.device)
        f_times_t = f_times_t / self.sample_rate
        
        # Compute sinc filters
        # ... (detailed sinc filter computation)
        
        # Apply filters to input signal
        return F.conv1d(x, filters, stride=1, padding=(self.kernel_size-1)//2)
```

### Key Features:
- Parameterized by low and high cutoff frequencies
- Frequencies are learned during training
- Uses Hamming window to reduce spectral leakage
- Provides interpretable frequency-selective filtering

## Component 2: Residual Blocks

Residual blocks form the backbone of the feature extraction process in RawNet2.

### Implementation Details:

```python
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            # Conv layer on the shortcut connection
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])
            self.lrelu1 = nn.LeakyReLU(negative_slope=0.3)
            
        # Main path
        self.conv1 = nn.Conv1d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv2 = nn.Conv1d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1
        )
        
        # Maxpooling on both paths
        self.mp = nn.MaxPool1d(3)
        
        # Shortcut connection
        self.shortcut = nn.Conv1d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=1,
            stride=1
        )
        
    def forward(self, x):
        if not self.first:
            # Process shortcut connection
            x_sc = self.bn1(x)
            x_sc = self.lrelu1(x_sc)
        else:
            x_sc = x
        
        # Shortcut connection
        x_sc = self.shortcut(x_sc)
        x_sc = self.mp(x_sc)
        
        # Main path
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.conv2(x)
        x = self.mp(x)
        
        # Add shortcut to main path
        x = x + x_sc
        
        return x
```

### Key Features:
- Skip connections to facilitate gradient flow
- Batch normalization for training stability
- LeakyReLU activation for non-linearity
- Max pooling for downsampling and feature selection

## Component 3: Frequency-Domain Attention Mechanism

The attention mechanism helps the model focus on the most relevant frequency components.

### Implementation Details:

```python
# Attention mechanism in the main model
def forward(self, x, is_test=False):
    # ... (previous layers)
    
    # Block 4 with attention
    x4 = self.block4(x)
    y4 = self.avgpool(x4).view(x4.size(0), -1)
    y4 = self.fc_attention4(y4)
    y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)
    x = x4 * y4 + y4  # Apply attention weights
    
    # Block 5 with attention
    x5 = self.block5(x)
    y5 = self.avgpool(x5).view(x5.size(0), -1)
    y5 = self.fc_attention5(y5)
    y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)
    x = x5 * y5 + y5  # Apply attention weights
    
    # ... (subsequent layers)
```

### Key Features:
- Global average pooling to capture channel-wise statistics
- Fully connected layer to compute attention weights
- Sigmoid activation to normalize weights between 0 and 1
- Multiplicative attention to emphasize important features

## Component 4: Gated Recurrent Unit (GRU)

The GRU layer models temporal dependencies in the extracted features.

### Implementation Details:

```python
# GRU implementation in the main model
def __init__(self, ...):
    # ... (other layers)
    
    # GRU layer
    self.gru = nn.GRU(
        input_size=nb_filts[3][1],
        hidden_size=gru_node,
        num_layers=nb_gru_layer,
        batch_first=True,
        bidirectional=False
    )
    
    # ... (other layers)

def forward(self, x, is_test=False):
    # ... (previous layers)
    
    # Prepare for GRU
    x = self.bn_before_gru(x)
    x = self.selu(x)
    x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
    
    # Apply GRU
    self.gru.flatten_parameters()
    x, _ = self.gru(x)
    x = x[:, -1, :]  # Take the last time step output
    
    # ... (subsequent layers)
```

### Key Features:
- Multiple GRU layers for hierarchical temporal modeling
- Batch-first processing for efficient computation
- Takes the last time step output as the sequence representation
- Flattens parameters for faster computation on GPU

## Component 5: Fully Connected Classification Layers

The final fully connected layers perform the classification task.

### Implementation Details:

```python
# FC layers in the main model
def __init__(self, ...):
    # ... (other layers)
    
    # FC layers after GRU
    self.fc1_gru = nn.Linear(in_features=gru_node, out_features=nb_fc_node)
    self.fc2_gru = nn.Linear(in_features=nb_fc_node, out_features=nb_classes)
    
    # ... (other layers)

def forward(self, x, is_test=False):
    # ... (previous layers)
    
    # Apply FC layers
    x = self.fc1_gru(x)
    x = self.fc2_gru(x)
    
    # Output handling
    if not is_test:
        output = x
        return output
    else:
        output = F.softmax(x, dim=1)
        return output
```

### Key Features:
- Two-layer fully connected network
- Different behavior during training and testing
- Softmax activation for probability output during testing
- Raw logits for loss computation during training

## Complete Model Architecture

The complete RawNet2 model architecture combines all these components in a sequential manner:

1. **Input Layer**: Raw audio waveform (64000 samples, mono)
2. **Sinc Filter Layer**: 128 filters for frequency-selective filtering
3. **Residual Blocks**: Multiple blocks organized in layers
   - Layer 1: 2 blocks with [128, 128] filters
   - Layer 2: 4 blocks with [128, 512] and [512, 512] filters
4. **Attention Mechanism**: Applied after blocks 4 and 5
5. **GRU Layer**: 3 layers with 1024 units each
6. **Fully Connected Layers**: 1024 nodes in hidden layer, 2 output classes

## Hyperparameters

The model uses the following hyperparameters for training:

```yaml
optimizer: Adam 
lr: 0.0001
wd: 0.0001
epoch: 100
batch_size: 32
seed: 1234
lr_decay: keras
amsgrad: 1
```

These components work together to create a powerful end-to-end architecture for audio deepfake detection that operates directly on raw waveforms.
