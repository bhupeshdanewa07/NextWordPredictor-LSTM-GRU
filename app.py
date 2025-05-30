import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Page configuration
st.set_page_config(
    page_title="LSTM Word Predictor | Bhupesh Danewa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #5D6D7E;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .creator-info {
        text-align: center;
        color: #E74C3C;
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 30px;
        padding: 10px;
        border: 2px solid #E74C3C;
        border-radius: 10px;
        background-color: #FADBD8;
    }
    .prediction-box {
        background-color: #D5DBDB;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
        margin: 20px 0;
    }
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1B4F72;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Load model and tokenizer with error handling
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load the LSTM Model
        model = load_model('next_word_lstm.h5')
        
        # Load the tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return model, tokenizer
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure 'next_word_lstm.h5' and 'tokenizer.pickle' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predicted[0])[-3:][::-1]
        predictions = []
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word, predictions
            if index in top_3_indices:
                confidence = predicted[0][index-1] * 100
                predictions.append((word, confidence))
        
        return None, predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, []

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 LSTM Next Word Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Text Completion using Deep Learning</p>', unsafe_allow_html=True)
    st.markdown('<div class="creator-info">🚀 Created by Bhupesh Danewa | AI/ML Enthusiast</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 About This Project")
        st.markdown("""
        This application uses **LSTM (Long Short-Term Memory)** neural networks 
        to predict the next word in a sequence. 
        
        **Features:**
        - Deep Learning powered predictions
        - Real-time text completion
        - Confidence scoring
        - Early stopping for optimal training
        
        **Built with:**
        - TensorFlow/Keras
        - Streamlit
        - NumPy
        """)
        
        st.markdown("### 👨‍💻 About Bhupesh Danewa")
        st.markdown("""
        Passionate about Machine Learning, Deep Learning, 
        and creating innovative AI solutions.
        
        Always exploring the frontiers of artificial intelligence!
        """)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Enter Your Text Sequence")
        input_text = st.text_input(
            "Type a sequence of words and let AI predict what comes next:",
            value="To be or not to",
            help="Enter a few words and the model will predict the most likely next word"
        )
        
        predict_button = st.button("🔮 Predict Next Word", use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Model Info")
        if model:
            max_sequence_len = model.input_shape[1] + 1
            st.metric("Max Sequence Length", max_sequence_len)
            st.metric("Model Input Shape", f"{model.input_shape}")
    
    # Prediction
    if predict_button and input_text.strip():
        with st.spinner("🤖 AI is thinking..."):
            time.sleep(0.5)  # Add a small delay for effect
            max_sequence_len = model.input_shape[1] + 1
            next_word, top_predictions = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            
            if next_word:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Predicted Next Word: **{next_word.upper()}**")
                st.markdown(f"**Complete Sentence:** *{input_text} {next_word}*")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show additional info
                st.markdown("### 📊 Prediction Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Input Length", len(input_text.split()))
                with col2:
                    st.metric("Tokens Processed", len(tokenizer.texts_to_sequences([input_text])[0]))
                with col3:
                    st.metric("Vocabulary Size", len(tokenizer.word_index))
            else:
                st.warning("⚠️ Could not predict the next word. Try a different input sequence.")
    
    elif predict_button:
        st.warning("📝 Please enter some text to get predictions!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #5D6D7E; padding: 20px;'>
        <b>LSTM Next Word Prediction App</b><br>
        Developed with ❤️ by <b>Bhupesh Danewa</b><br>
        <i>Exploring the possibilities of AI and Natural Language Processing</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()