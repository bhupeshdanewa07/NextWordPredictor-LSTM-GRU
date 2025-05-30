# 🧠 LSTM Next Word Prediction

A sophisticated web application that predicts the next word in a text sequence using **LSTM (Long Short-Term Memory)** neural networks. Features early stopping for optimal training and a modern Streamlit interface.

**Created by: Bhupesh Danewa**

## ✨ Features

- 🤖 Deep Learning powered next word prediction
- 🎯 Real-time text completion with confidence scoring
- 📊 Interactive model analytics and metrics
- 🎨 Professional UI with responsive design
- ⚡ Early stopping mechanism to prevent overfitting

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install streamlit tensorflow numpy pickle-mixin
   ```

2. **Ensure model files are present**
   - `next_word_lstm.h5` (trained LSTM model)
   - `tokenizer.pickle` (text tokenizer)

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open browser** at `http://localhost:8501`

## 🎯 Usage

1. Enter a text sequence in the input field (e.g., "To be or not to")
2. Click "Predict Next Word" to get AI-powered predictions
3. View the predicted word along with model analytics

## 🔧 Model Performance & Improvements

The current model uses early stopping for optimal training, but **significant accuracy improvements** can be achieved by:

### Training with More Epochs
```python
# Train for more epochs to improve accuracy
model.fit(X, y, epochs=100, batch_size=64, 
          validation_split=0.2, 
          callbacks=[early_stopping], 
          verbose=1)
```

### Potential Enhancements
- **Increase training epochs** (50-200 epochs) for better convergence
- **Larger training dataset** for more diverse vocabulary
- **Hyperparameter tuning** (LSTM units, learning rate, batch size)
- **Advanced architectures** (Bidirectional LSTM, GRU, or Transformer models)

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── next_word_lstm.h5      # Trained LSTM model
├── tokenizer.pickle       # Text tokenizer
└── README.md             # Documentation
```

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **Streamlit** - Web application framework  
- **NumPy** - Numerical computing
- **LSTM** - Long Short-Term Memory networks

---
*Built with ❤️ by Bhupesh Danewa | AI/ML Enthusiast*
