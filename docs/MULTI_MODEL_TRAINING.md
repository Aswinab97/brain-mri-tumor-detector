# Multi-Model Training for Improved Accuracy

This document describes the enhanced multi-model training capabilities added to improve the Brain MRI Tumor Detection accuracy.

## Overview

The project now supports training and using multiple state-of-the-art deep learning models:

1. **ResNet18** - Deep residual network (baseline)
2. **VGG16** - Visual Geometry Group network with 16 layers
3. **DenseNet121** - Densely connected convolutional network
4. **EfficientNetB0** - Efficient architecture with compound scaling
5. **MobileNetV2** - Lightweight model optimized for mobile/edge devices

## Key Features

### 1. Multi-Model Training (`src/train.py`)
- Train multiple pretrained models with transfer learning
- Automated data augmentation for better generalization
- Model evaluation with comprehensive metrics
- Checkpoint saving for each model

### 2. Ensemble Prediction (`src/inference.py`)
- Support for single model or ensemble prediction
- Ensemble averaging across multiple models for improved accuracy
- Automatic model loading and fallback handling
- Detailed prediction information including per-model probabilities

### 3. Model Comparison (`src/compare_models.py`)
- Evaluate and compare all trained models
- Generate comprehensive comparison reports
- Identify best performing model
- Save metrics for each model

### 4. Enhanced API (`src/api.py`)
- New `/models` endpoint to check loaded models
- Updated `/predict` endpoint with ensemble support
- Detailed prediction responses with model information
- Backward compatible with single model deployment

## Training New Models

### Prerequisites

1. Organize your data in the following structure:
```
data_raw/
├── yes/      # MRI images with tumors
└── no/       # MRI images without tumors
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Process

1. **Train multiple models:**
```bash
python -m src.train
```

This will train all 5 models and save checkpoints to the `models/` directory.

2. **Compare model performance:**
```bash
python -m src.compare_models
```

This generates comparison reports in the `reports/` directory.

### Expected Model Files

After training, you should have:
```
models/
├── resnet18_brain_mri.pth
├── vgg16_brain_mri.pth
├── densenet121_brain_mri.pth
├── efficientnet_b0_brain_mri.pth
└── mobilenet_v2_brain_mri.pth
```

## Using Trained Models

### Ensemble Prediction (Recommended)

When multiple models are available, the API automatically uses ensemble prediction:

```python
# All models in models/ directory will be loaded
# Predictions are averaged across all models
```

### Single Model Prediction

To use a specific model, you can modify the API initialization:

```python
classifier = BrainTumorClassifier(
    model_paths={"resnet18": "models/resnet18_brain_mri.pth"},
    use_ensemble=False
)
```

## API Endpoints

### Check Loaded Models
```bash
GET /models
```

Response:
```json
{
  "loaded_models": ["resnet18", "vgg16", "densenet121"],
  "ensemble_mode": true,
  "prediction_type": "ensemble"
}
```

### Make Prediction
```bash
POST /predict
```

Response with ensemble:
```json
{
  "filename": "brain_mri.jpg",
  "label": 1,
  "label_name": "tumor",
  "probability": 0.87,
  "prediction_type": "ensemble",
  "model_predictions": {
    "resnet18": 0.89,
    "vgg16": 0.85,
    "densenet121": 0.87
  }
}
```

## Expected Improvements

With the multi-model ensemble approach, we expect:

- **Higher Accuracy**: Ensemble typically outperforms individual models by 2-5%
- **Better Generalization**: Multiple architectures capture different features
- **Robust Predictions**: Averaging reduces impact of individual model errors
- **Flexibility**: Can use best single model or ensemble based on requirements

## Model Performance Comparison

| Model | Parameters | Inference Speed | Expected Accuracy |
|-------|-----------|-----------------|-------------------|
| ResNet18 | 11M | Fast | ~89-92% |
| VGG16 | 138M | Slow | ~87-90% |
| DenseNet121 | 8M | Medium | ~90-93% |
| EfficientNetB0 | 5M | Fast | ~91-94% |
| MobileNetV2 | 3.5M | Very Fast | ~86-89% |
| **Ensemble** | Combined | Slower | **~92-95%** |

*Note: Actual performance depends on your specific dataset*

## Data Augmentation

The training script includes the following augmentation techniques:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness and contrast)
- Standard ImageNet normalization

## Deployment Considerations

### For Production
- Use ensemble for highest accuracy (recommended)
- Requires more memory and compute resources
- Slower inference but better results

### For Edge/Mobile
- Use MobileNetV2 or EfficientNetB0
- Faster inference with lower resource usage
- Slightly lower accuracy but still effective

### For Balance
- Use ResNet18 or DenseNet121
- Good balance of accuracy and speed
- Suitable for most applications

## Troubleshooting

### No Models Found
If the API shows "dummy" prediction type:
1. Check that model files exist in `models/` directory
2. Verify file names match expected format
3. Ensure models are properly trained and saved

### Out of Memory
If training runs out of memory:
1. Reduce batch size in training script
2. Train models one at a time
3. Use smaller models (MobileNetV2, EfficientNetB0)

### Low Accuracy
To improve accuracy:
1. Add more training data
2. Increase training epochs
3. Try different learning rates
4. Use stronger data augmentation

## References

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- EfficientNet: [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- MobileNetV2: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
