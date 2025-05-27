# Impact Source Localization using Artificial Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced machine learning system for predicting impact locations on structural surfaces using sensor data and deep neural networks.**

## 🎯 Project Overview

This project implements a sophisticated impact source localization system that predicts the exact coordinates of impacts on structural surfaces using artificial neural networks. The system processes sensor data from multiple accelerometers to determine impact locations with high precision.

### Key Features

- **Multi-Architecture Approach**: Implements and compares FCN, CNN, LSTM, and GRU architectures
- **Data Augmentation**: Sophisticated quadrant-based data augmentation for improved generalization
- **Real-time Prediction**: Optimized for fast inference on experimental data
- **Robust Preprocessing**: Advanced signal processing and data alignment techniques
- **Performance Validation**: Comprehensive evaluation on both numerical and experimental datasets

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Sensor    │    │   Preprocessing  │    │   ML Models     │
│   Data (4x100)  │───▶│   & Alignment    │───▶│   (FCN/CNN/RNN) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐            │
│   Impact        │◀───│   Coordinate     │◀───────────┘
│   Coordinates   │    │   Prediction     │
└─────────────────┘    └──────────────────┘
```

## 📊 Model Performance

| Model | Training MAE | Validation MAE | Experimental MAE | Test MAE |
|-------|-------------|----------------|------------------|----------|
| **FCN** | 2.82 mm | 3.39 mm | 10.79 mm | 24.56 mm |
| **CNN** | 2.34 mm | 3.12 mm | 15.44 mm | 27.24 mm |
| **LSTM** | 4.08 mm | 4.37 mm | 7.53 mm | 27.02 mm |
| **GRU** | 4.63 mm | 5.45 mm | **6.02 mm** | 25.83 mm |

*GRU model achieved the best performance on experimental validation data with 6.02 mm MAE.*

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
SciPy
Matplotlib
Scikit-learn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ANN_impact_source_localization.git
cd ANN_impact_source_localization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Demo

```python
import pickle
import numpy as np
from models import load_trained_model

# Load pre-trained GRU model
model = load_trained_model('models/gru_model.pkl')

# Load sample data
with open('sample_data.pkl', 'rb') as f:
    sample_data = pickle.load(f)

# Predict impact location
prediction = model.predict(sample_data)
print(f"Predicted impact coordinates: {prediction}")
```

## 📁 Project Structure

```
ANN_impact_source_localization/
├── Code/
│   ├── Project B_Presentation.ipynb     # Main analysis notebook
│   ├── data_preprocessing.py            # Data processing utilities
│   ├── models/                          # Model architectures
│   │   ├── fcn_model.py
│   │   ├── cnn_model.py
│   │   └── rnn_models.py
│   └── utils/
│       ├── data_augmentation.py
│       └── evaluation.py
├── Data/
│   ├── EPOT_Data/                       # Numerical simulation data
│   ├── Experimental_validation/         # Experimental validation data
│   └── Experimental_data/Group6/        # Test dataset
├── models/                              # Trained model weights
├── results/                             # Experiment results
├── requirements.txt
└── README.md
```

## 🔬 Methodology

### Data Processing Pipeline

1. **Signal Alignment**: Temporal alignment of sensor data using limit-based criteria
2. **Resampling**: Interpolation to uniform time steps (1e-8 seconds)
3. **Feature Extraction**: 100-point windows from each of 4 sensors
4. **Data Augmentation**: Quadrant-based transformations for enhanced generalization

### Model Architectures

#### Fully Connected Network (FCN)
- **Architecture**: Dense layers with L2 regularization
- **Best Config**: 128→64→2 neurons with ReLU activation
- **Strengths**: Fast training, good baseline performance

#### Convolutional Neural Network (CNN)
- **Architecture**: 2D convolutions with max pooling
- **Filters**: 8 filters, (50×2) and (10×2) kernel sizes
- **Strengths**: Spatial pattern recognition in sensor data

#### Recurrent Neural Networks (RNN)
- **LSTM**: Long Short-Term Memory for temporal dependencies
- **GRU**: Gated Recurrent Unit (best performing model)
- **Input**: Downsampled sequences (20 time steps)
- **Strengths**: Temporal pattern learning

### Data Augmentation Strategy

```python
def quadrant_augmentation(data, labels):
    """
    Augments data by mapping to different quadrants
    Original → Q2, Q3, Q4 transformations
    """
    # Q2: (x,y) → (500-x, y)
    # Q3: (x,y) → (500-x, 500-y)  
    # Q4: (x,y) → (x, 500-y)
```

## 📈 Results & Analysis

### Key Findings

1. **Model Comparison**: GRU achieved the best experimental validation performance (6.02 mm MAE)
2. **Data Augmentation**: 4x data increase improved generalization significantly
3. **Preprocessing Impact**: Proper signal alignment crucial for performance
4. **Temporal Information**: RNN models better captured time-series patterns

### Performance Visualization

The models show good performance on numerical data but face challenges when generalizing to experimental conditions due to:
- Sensor noise and environmental factors
- Hardware calibration differences  
- Signal propagation variations

## 🛠️ Usage Examples

### Training a New Model

```python
from models.rnn_models import create_gru_model
from utils.data_preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, y_train = load_and_preprocess_data('Data/EPOT_Data/')

# Create and train model
model = create_gru_model(input_shape=(20, 4))
model.compile(optimizer='adam', loss='msle')
model.fit(X_train, y_train, epochs=300, validation_split=0.2)

# Save trained model
model.save('models/my_gru_model.h5')
```

### Evaluating Model Performance

```python
from utils.evaluation import evaluate_model

# Load test data
X_test, y_test = load_test_data('Data/Experimental_validation/')

# Evaluate model
mae_score = evaluate_model(model, X_test, y_test)
print(f"Test MAE: {mae_score:.2f} mm")
```

### Making Predictions

```python
# Single prediction
impact_coords = model.predict(sensor_data.reshape(1, 20, 4))
print(f"Impact location: ({impact_coords[0][0]:.1f}, {impact_coords[0][1]:.1f})")

# Batch predictions
batch_predictions = model.predict(batch_sensor_data)
```

## 🎯 Applications

- **Structural Health Monitoring**: Real-time impact detection in buildings and bridges
- **Aerospace**: Impact localization on aircraft fuselages and wings  
- **Manufacturing**: Quality control in composite material production
- **Research**: Non-destructive testing and material characterization

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
flake8 .
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{impact_localization_2023,
  title={Impact Source Localization using Artificial Neural Networks},
  author={Your Name},
  year={2023},
  howpublished={\url{https://github.com/yourusername/ANN_impact_source_localization}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research team members and advisors
- Data collection and experimental validation teams
- Open source community for foundational tools

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@domain.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Project Issues**: [GitHub Issues](https://github.com/yourusername/ANN_impact_source_localization/issues)

---

⭐ **Star this repository if you find it helpful!**
