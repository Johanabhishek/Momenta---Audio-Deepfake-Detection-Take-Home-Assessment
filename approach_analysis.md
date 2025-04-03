# Analysis of Audio Deepfake Detection Approaches

## Evaluation Criteria
- Effectiveness in detecting AI-generated human speech
- Potential for real-time or near real-time detection
- Suitability for analyzing real conversations
- Performance metrics (EER, t-DCF)
- Computational efficiency

## Handcrafted Feature-based Approaches

### 1. Voice spoofing countermeasure for logical access attacks detection
- **Feature Extraction**: ELTP-LFCC
- **Network Structure**: DBiLSTM
- **Performance**: LA: 0.74% (1), t-DCF: 0.008 (1)
- **Real-time Potential**: High - LFCC features are computationally efficient
- **Notes**: Excellent performance metrics, ranked #1 in LA scenario

### 2. Voice spoofing detector: A unified anti-spoofing framework
- **Feature Extraction**: ATP-GTCC
- **Network Structure**: SVM
- **Performance**: LA: 0.75% (2), PA: 1.00% (1), t-DCF: 0.050 (2)
- **Real-time Potential**: High - SVM is lightweight for inference
- **Notes**: Well-balanced performance across LA and PA scenarios

### 3. Detecting spoofing attacks using VGG and SincNet
- **Feature Extraction**: CQT, Power Spectrum
- **Network Structure**: VGG, SincNet
- **Performance**: LA: 8.01% (4), PA: 1.51% (2), t-DCF: 0.208 (4)
- **Real-time Potential**: Medium - VGG networks can be computationally intensive
- **Notes**: Good performance on physical access attacks

## Hybrid Feature-based Approaches

### 4. Light convolutional neural network with feature genuinization
- **Feature Extraction**: CQT-based LPS
- **Network Structure**: LCNN
- **Performance**: LA: 4.07% (11), t-DCF: 0.102 (10)
- **Real-time Potential**: High - Light CNN architecture designed for efficiency
- **Notes**: Good balance between performance and computational efficiency

### 5. Generalization of audio deepfake detection
- **Feature Extraction**: LFB
- **Network Structure**: ResNet18
- **Performance**: LA: 1.81% (4), t-DCF: 0.052 (4)
- **Real-time Potential**: Medium - ResNet18 is relatively efficient but still demanding
- **Notes**: Good generalization capabilities for detecting various deepfake types

### 6. Siamese convolutional neural network using gaussian probability feature
- **Feature Extraction**: LFCC
- **Network Structure**: Siamese CNN
- **Performance**: LA: 3.79% (10), PA: 7.98% (5), t-DCF: 0.093 (5)
- **Real-time Potential**: Medium - Siamese networks require paired comparisons
- **Notes**: Innovative approach using probability features

## End-to-end Approaches

### 7. Raw differentiable architecture search for speech deepfake and spoofing detection
- **Feature Extraction**: Sinc Filter
- **Network Structure**: PC-DARTS
- **Performance**: LA: 1.77% (10), t-DCF: 0.052 (7)
- **Real-time Potential**: Low - Architecture search is computationally intensive
- **Notes**: Automated architecture design may provide better generalization

### 8. End-to-end anti-spoofing with RawNet2
- **Feature Extraction**: Sinc Filter
- **Network Structure**: RawNet2
- **Performance**: LA: 1.12% (5), t-DCF: 0.033 (3)
- **Real-time Potential**: Medium - Works directly on raw waveforms
- **Notes**: Excellent performance metrics, works on raw audio without preprocessing

### 9. Towards end-to-end synthetic speech detection
- **Feature Extraction**: DNN
- **Network Structure**: Res-TSSDNet, Inc-TSSDNet
- **Performance**: LA: 1.64% (9), t-DCF: 0.048 (6)
- **Real-time Potential**: Medium - Specialized architecture for speech detection
- **Notes**: Specifically designed for synthetic speech detection

## Feature Fusion-based Approaches

### 10. Betray oneself: A novel audio deepfake detection model via mono-to-stereo conversion
- **Feature Extraction**: Sinc Filter
- **Network Structure**: AASIST, M2S Converter
- **Performance**: LA: 1.34% (1)
- **Real-time Potential**: Medium - Novel approach with mono-to-stereo conversion
- **Notes**: Innovative approach that may detect subtle artifacts in deepfakes

### 11. Detection of cross-dataset fake audio based on prosodic and pronunciation features
- **Feature Extraction**: Phoneme Feature, Prosody Feature, Wav2Vec2
- **Network Structure**: LCNN, Bi-LSTM
- **Performance**: LA: 1.58% (3)
- **Real-time Potential**: Medium - Linguistic feature extraction may add latency
- **Notes**: Good for cross-dataset generalization, focuses on linguistic artifacts

### 12. Overlapped frequency-distributed network: Frequency-aware voice spoofing countermeasure
- **Feature Extraction**: Mel-Spectrogram, CQT
- **Network Structure**: LCNN, ResNet
- **Performance**: LA: 1.35% (2), PA: 0.35
- **Real-time Potential**: Medium - Multiple feature extraction adds complexity
- **Notes**: Frequency-aware approach may detect subtle spectral artifacts
