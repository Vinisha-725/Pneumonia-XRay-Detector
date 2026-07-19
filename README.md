# Pneumonia Chest X-Ray Detector

A deep learning application that detects pneumonia from chest X-ray images using DenseNet121 transfer learning.

## Features

- **Pneumonia Detection**: Classifies chest X-rays as NORMAL or PNEUMONIA
- **Grad-CAM Visualization**: Shows which regions of the image contributed to the model's decision
- **Interactive UI**: Built with Streamlit for easy image upload and real-time predictions
- **High Accuracy**: Achieves 92.31% accuracy on test data

## Model Performance

- **Accuracy**: 92.31%
- **Precision**: 90.33%
- **Recall**: 98.21%
- **F1 Score**: 94.10%
- **AUC-ROC**: 97.66%

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/Vinisha-725/Pneumonia-XRay-Detector.git
cd Pneumonia-XRay-Detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Requirements

- Python 3.8+
- streamlit
- numpy
- Pillow
- opencv-python-headless
- tensorflow
- tf-nightly

## Project Structure

```
Pneumonia-XRay-Detector/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── densenet121_pneumonia.keras # Pre-trained DenseNet121 model
├── Chest_XRAY.ipynb           # Training notebook
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
└── README.md                   # This file
```

## How It Works

1. **Image Upload**: Upload a chest X-ray image (JPG, JPEG, or PNG)
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **Prediction**: DenseNet121 model processes the image and outputs a probability
4. **Classification**: 
   - Probability > 0.5 → PNEUMONIA
   - Probability ≤ 0.5 → NORMAL
5. **Visualization**: Grad-CAM heatmap shows model attention regions

## Dataset

The model was trained on the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset.

- **Classes**: NORMAL, PNEUMONIA
- **Total Images**: ~5,800
- **Split**: Training, Validation, Test

## Model Architecture

**DenseNet121** (pre-trained on ImageNet)

- **Phase 1**: Backbone frozen, only classifier trained
- **Phase 2**: Entire network unfrozen and fine-tuned with lower learning rate

DenseNet connects each layer to every subsequent layer, improving:
- Feature reuse
- Gradient flow
- Parameter efficiency

## Deployment

This app is deployed on Streamlit Cloud at:
[https://pneomonia-detector-vs.streamlit.app/](https://pneomonia-detector-vs.streamlit.app/)

## Limitations

- Model performance depends on image quality
- Should be used as a screening tool, not for definitive diagnosis
- Always consult medical professionals for health decisions

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: Kaggle Chest X-Ray Images (Pneumonia)
- Model Architecture: DenseNet (Huang et al., 2017)
- Framework: TensorFlow/Keras, Streamlit
