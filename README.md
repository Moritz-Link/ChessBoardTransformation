# ChessBoardTransformation

A deep learning project for chess board corner detection and perspective transformation, enabling conversion from rotated chess board views to bird's eye perspective using keypoint detection.

## Project Overview

This project develops a **keypoint detection model** that identifies the four corners of a chess board in images. The detected keypoints can then be used for perspective transformation to convert rotated or angled chess board views into a standardized bird's eye view perspective.

### Key Features

- **Custom CNN Architecture**: Purpose-built neural network for chess board corner detection
- **Custom Dataset**: Specially created dataset with annotated chess board keypoints
- **4-Point Detection**: Identifies the four corners of chess boards with high precision
- **Perspective Transformation Ready**: Output coordinates can be directly used for geometric transformations
- **Data Augmentation**: Robust training with various image transformations

## Methodology

### Model Architecture

The project uses a **custom Convolutional Neural Network (CNN)** architecture rather than traditional YOLO, specifically designed for keypoint regression:

- **Input**: RGB images (resized to 312×312 pixels)
- **Output**: 8 coordinates (4 keypoints × 2 coordinates each: x,y)
- **Architecture**: 4 convolutional layers + 3 fully connected layers
- **Features**: Batch normalization, dropout regularization, ELU activation

```
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.1)
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.2)
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.3)
Conv2D(256) → BatchNorm → MaxPool → Dropout(0.4)
Flatten → FC(1000) → BatchNorm → Dropout(0.5)
FC(500) → BatchNorm → Dropout(0.6)
FC(8)  # Final layer outputs 8 coordinates
```

### Dataset Creation

The project includes a **custom-created dataset** with the following characteristics:

- **Annotated Chess Boards**: Images with precisely labeled corner coordinates
- **Bounding Box Information**: Complete board boundary annotations
- **Multiple Formats**: Train/validation/test splits with JSON metadata
- **Data Augmentation**: Vertical flips, brightness/contrast variations, resizing

Dataset format includes:
- Image paths and dimensions
- 4 keypoint coordinates (corners of chess board)
- Bounding box coordinates
- Comprehensive metadata for each image

### Training Process

- **Loss Function**: Mean Squared Error (MSE) for coordinate regression
- **Optimizer**: Adam optimizer with learning rate 0.001
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Evaluation Metrics**: MSE and MAE per keypoint
- **Batch Processing**: Configurable batch sizes for training/validation/testing

## Repository Structure

```
ChessBoardTransformation/
├── README.md                    # This file
├── KeyPointModelDataset.py      # Dataset class and CNN model definition
├── TrainingHelper.py            # Training utilities and evaluation functions
├── ChessBoardTemplate.jpg       # Template chess board image
│
├── Dataset Files/
│   ├── TrainData_json.json      # Training dataset metadata
│   ├── ValidData_json.json      # Validation dataset metadata
│   ├── TestData_json.json       # Test dataset metadata
│   ├── TotalFinalData_json.json # Complete dataset metadata
│   ├── train.zip               # Training images
│   ├── test.zip                # Test images
│   └── valid.zip               # Validation images
│
├── Results/                     # Training results and visualizations
│   ├── 2_results.png
│   ├── 3_results.png
│   └── 5_results.png
│
└── Settings/                    # Training configuration files
    ├── 4_Setting.json
    └── 5_Setting.json
```

## Usage

### Requirements

```python
torch
torchvision
opencv-python (cv2)
matplotlib
numpy
pandas
albumentations
Pillow
beautifulsoup4
```

### Basic Usage

1. **Load the Dataset**:
```python
from KeyPointModelDataset import KeyPointDetectionDS

# Initialize dataset
dataset = KeyPointDetectionDS(
    path2json='TrainData_json.json',
    file_base_path='path/to/images',
    type="train",
    transform=True
)
```

2. **Initialize the Model**:
```python
from KeyPointModelDataset import Net

# Create model instance
model = Net()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

3. **Training**:
```python
from TrainingHelper import train, EarlyStopping
from torch.optim import Adam
from torch.nn import MSELoss

# Setup training parameters
args = {
    "TrainDataset": train_dataset,
    "ValDataset": val_dataset,
    "TestDataset": test_dataset,
    "TrainBatchSize": 5,
    "ValBatchSize": 5,
    "TestBatchSize": 5,
    "EPOCHS": 20,
    "model": model,
    "LossFunction": MSELoss(),
    "Optimizer": Adam,
    "Optimizer_LR": 0.001,
    "EarlyStopping": EarlyStopping(patience=7),
    "StartEarlyStopping": 18,
    "RUN": "experiment_1"
}

# Train the model
loss_dict, train_params = train(args)
```

### Data Format

The dataset JSON files contain entries in the following format:

```json
{
  "0": {
    "image_path": "test/image.jpg",
    "height": "1371",
    "width": "2048",
    "points_points": "350.80,108.96;182.11,1220.78;1573.04,121.22;1864.42,1196.25",
    "box_label": "board",
    "xtl": "171.37",
    "ytl": "105.70",
    "xbr": "1881.30",
    "ybr": "1236.12"
  }
}
```

Where:
- `points_points`: Four corner coordinates in format "x1,y1;x2,y2;x3,y3;x4,y4"
- `xtl, ytl, xbr, ybr`: Bounding box coordinates (top-left, bottom-right)
- `height, width`: Image dimensions

## Results

The model successfully detects chess board corners with high accuracy. Training results show:

- **Convergence**: Model converges within 20 epochs with early stopping
- **Metrics**: Low MSE and MAE per keypoint on validation data
- **Robustness**: Handles various chess board orientations and lighting conditions

Example results are visualized in the `Results/` directory showing:
- Training loss curves
- Validation metrics over time
- Model performance analysis

## Applications

The detected keypoints can be used for:

1. **Perspective Transformation**: Convert rotated boards to bird's eye view
2. **Chess Position Analysis**: Standardize board orientation for piece recognition
3. **Augmented Reality**: Overlay digital content on physical chess boards
4. **Board Tracking**: Monitor chess board state in video streams

## Technical Details

### Model Performance
- **Input Size**: 312×312 RGB images
- **Output**: 8 floating-point coordinates
- **Training Time**: Approximately 20 epochs with early stopping
- **Hardware**: Optimized for both CPU and GPU training

### Data Augmentation
- Resize to 312×312 pixels
- Vertical flips (40% probability)
- Random brightness/contrast adjustments (20% probability)
- Maintains keypoint coordinate consistency

### Evaluation Metrics
- **MSE (Mean Squared Error)**: Primary loss function
- **MAE (Mean Absolute Error)**: Additional evaluation metric
- **Per-keypoint accuracy**: Individual corner detection precision

## Future Enhancements

Potential improvements and extensions:

- [ ] Integration with piece detection models
- [ ] Real-time video processing capabilities
- [ ] Support for different board sizes and styles
- [ ] Mobile deployment optimization
- [ ] Integration with chess engines for position analysis

## Citation

If you use this work in your research, please cite:

```
@misc{chessBoardTransformation2024,
  title={Chess Board Transformation: Keypoint Detection for Perspective Correction},
  author={[Author Name]},
  year={2024},
  url={https://github.com/Moritz-Link/ChessBoardTransformation}
}
```

## License

This project is available under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.