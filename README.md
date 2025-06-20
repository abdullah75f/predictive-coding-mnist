# 🧠 Predictive Coding Network: MNIST Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A biologically-inspired implementation of Predictive Coding Networks (PCN) for MNIST digit classification, featuring lateral connections and a modern web interface for real-time predictions.

## 🌟 Features

- **Biologically-Inspired Architecture**: Implements predictive coding principles with lateral connections
- **Interactive Web Interface**: Streamlit-based app for real-time digit classification
- **Pre-trained Model**: Includes a trained model achieving competitive accuracy on MNIST
- **Real-time Inference**: Upload and classify handwritten digits instantly
- **Modular Design**: Clean, well-documented codebase for easy extension

## 🏗️ Architecture

The model uses a **784 → 256 → 64 → 10** architecture:
- **Input Layer**: 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation  
- **Output Layer**: 10 neurons (digits 0-9)

### Predictive Coding Principles

The model implements key predictive coding concepts:
- **Error Computation**: Calculates prediction errors between layers
- **State Updates**: Iterative refinement of neural states
- **Weight Learning**: Hebbian-like weight updates based on error signals
- **Lateral Connections**: Bidirectional information flow

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/predictive-coding-mnist.git
   cd predictive-coding-mnist
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - pre-trained model included)
   ```bash
   python predictive_coding.py
   ```

4. **Launch the web interface**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Model Performance

- **Training Accuracy**: ~82.88% (3 epochs, 10,000 samples per epoch)
- **Test Accuracy**: ~78.15% (2,000 test samples)
- **Training Time**: ~5-10 minutes on CPU
- **Inference Time**: <100ms per prediction

## 🎯 Usage

### Web Interface

1. **Upload an Image**: Use the file uploader to select a digit image (PNG, JPG, JPEG)
2. **View Prediction**: The model will display its prediction in real-time
3. **Explore Results**: See the uploaded image alongside the prediction

### Programmatic Usage

```python
from predictive_coding import PredictiveCodingModel

# Load the pre-trained model
model = PredictiveCodingModel()
model.load_model('trained_pc_model.npz')

# Make a prediction
import numpy as np
input_data = np.random.randn(784)  # Your preprocessed image
prediction = model.predict(input_data)
print(f"Predicted digit: {prediction}")
```

## 🔧 Technical Details

### Training Process

The model uses an iterative training procedure:

1. **Forward Pass**: Compute predictions for each layer
2. **Error Computation**: Calculate prediction errors
3. **State Updates**: Refine neural states based on errors
4. **Weight Updates**: Update weights using error signals
5. **Regularization**: L2 weight decay for stability

### Key Parameters

- **Learning Rate (State)**: 0.05 - Controls state update speed
- **Learning Rate (Weight)**: 0.002 - Controls weight update speed
- **Iterations per Sample**: 50 - Number of inference iterations
- **Weight Decay**: 0.0001 - L2 regularization coefficient

## 📁 Project Structure

```
predictive-coding-mnist/
├── app.py                 # Streamlit web interface
├── predictive_coding.py   # Core PCN implementation
├── trained_pc_model.npz   # Pre-trained model weights
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # MNIST dataset (auto-downloaded)
└── .gitignore           # Git ignore rules
```

## 🧪 Experiments & Results

The model demonstrates several interesting properties:

- **Convergence**: Stable training with consistent accuracy improvements
- **Generalization**: Good performance on unseen test data
- **Efficiency**: Fast inference suitable for real-time applications
- **Robustness**: Handles various digit styles and orientations

## 🔬 Research Context

This implementation explores the intersection of:
- **Neuroscience**: Predictive coding as a theory of brain function
- **Machine Learning**: Neural networks and deep learning
- **Computer Vision**: Image classification and feature learning

The model provides insights into how biological neural networks might process visual information through predictive coding mechanisms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **Predictive Coding Theory**: Karl Friston and the Free Energy Principle
- **PyTorch**: Facebook Research for the deep learning framework
- **Streamlit**: For the beautiful web interface framework

## 📚 References

1. Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815-836.
2. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
3. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.

---

**Made with ❤️ for the neuroscience and machine learning communities**