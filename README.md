# 🧠 Predictive Coding Network: MNIST Classifier

A biologically-inspired neural network implementation for MNIST digit classification using predictive coding principles.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the web app**
   ```bash
   streamlit run app.py
   ```

3. **Open browser** at `http://localhost:8501`

## Architecture

**Network Structure**: 784 → 256 → 64 → 10 neurons
- **Input**: 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output**: 10 neurons (digits 0-9)

## What it does

- **Upload** handwritten digit images (PNG, JPG, JPEG)
- **Predict** the digit (0-9) using a predictive coding network
- **Performance**: ~78% accuracy on MNIST test set

## Files

- `app.py` - Streamlit web interface
- `predictive_coding.py` - Core model implementation
- `trained_pc_model.npz` - Pre-trained model weights
- `requirements.txt` - Python dependencies

## How it works

The model uses predictive coding principles:
1. **Forward pass** - Generate predictions for each layer
2. **Error computation** - Calculate prediction errors
3. **State updates** - Refine neural states iteratively
4. **Weight learning** - Update weights based on errors

## Train your own model

```bash
python predictive_coding.py
```

This will train for 3 epochs on 10,000 MNIST samples and save the model.