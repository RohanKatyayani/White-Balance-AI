# AI White Balance Prediction for Professional Photography

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Computer Vision](https://img.shields.io/badge/Computer-Vision-green)
![M4 Mac](https://img.shields.io/badge/M4-Mac_Optimized-purple)

An end-to-end machine learning solution that predicts professional white balance settings (Temperature & Tint) from RAW camera images and metadata.

## 🎯 Problem Statement

Professional photographers spend hours manually adjusting white balance settings. This AI automates that process by learning a photographer's editing style and predicting optimal Temperature (2000-50000K) and Tint (-150 to +150) values.

## 🏗️ Solution Architecture

### Data Pipeline
- **2538 professional RAW images** with expert-edited white balance labels
- **Camera metadata integration**: Aperture, ISO, Focal Length, Shutter Speed
- **As Shot White Balance**: Uses camera's initial WB as reference point
- **TIFF image processing** with proper normalization

### Model Evolution
1. **Regression Models** - Direct value prediction (challenging due to wide ranges)
2. **Adjustment Prediction** - Predict changes from camera's As Shot WB
3. **🎯 Classification Approach** - Predict temperature/tint ranges (most effective)

### Final Model
- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Input**: 224x224 images + 8 metadata features
- **Output**: Temperature range (10 classes) + Tint range (9 classes)
- **Accuracy**: 49% Temperature, 73% Tint

## 📊 Key Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Wide value ranges (2000-50000K) | Classification into bins |
| Non-linear temperature sensitivity | Custom weighted loss function |
| Consistency across similar images | Metadata integration + data augmentation |
| M4 Mac optimization | MPS acceleration + efficient data loading |

## 🚀 Installation & Usage

  # Clone repository
  git clone https://github.com/yourusername/white-balance-ai.git
  cd white-balance-ai
  
  # Install dependencies
  pip install torch torchvision pandas numpy opencv-python pillow matplotlib
  
  # Run training
  python train.py
  
  # Make predictions
  python predict.py --image_path path/to/image.tif


## 🎨 Technical Highlights
Multi-modal Learning: Combines image features + camera metadata

Production Ready: Handles real-world camera RAW formats

Apple Silicon Optimized: Full MPS support for M1/M2/M4 Macs

Non-destructive Editing: Predicts adjustments without altering originals

📈 Results
Metric	Performance
Temperature Accuracy	49.1%
Tint Accuracy	73.4%
Training Time	~2 minutes/epoch (M4 Mac)
Inference Speed	~50ms/image
🛠️ Technologies Used
PyTorch - Deep learning framework

TorchVision - Pretrained models & transforms

Pandas - Data processing

OpenCV/PIL - Image handling

Apple MPS - GPU acceleration

🎯 Business Impact
This solution addresses a real pain point in professional photography workflows:

Reduces editing time from hours to seconds

Maintains consistent editing style

Handles diverse lighting conditions

Works across different camera brands

🔮 Future Improvements
Fine-grained temperature bins for better precision

Ensemble methods for improved consistency

Temporal consistency for video sequences

User feedback loop for continuous learning

👨‍💻 Author
Rohan Katyayani - [Your GitHub Profile]

Built as part of a real-world AI competition, demonstrating end-to-end machine learning capabilities from problem understanding to deployment-ready solution.

⭐ Star this repo if you found it helpful!
