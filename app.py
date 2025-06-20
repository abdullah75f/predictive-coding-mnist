import streamlit as st
import numpy as np
from PIL import Image
from predictive_coding import PredictiveCodingModel

# Set the page title and icon
st.set_page_config(page_title="PCN Classifier", page_icon="🧠")

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Load the predictive coding model from the saved file."""
    model = PredictiveCodingModel()
    try:
        model.load_model('trained_pc_model.npz')
    except FileNotFoundError:
        # Provide a helpful error message if the model file doesn't exist.
        st.error("Model file 'trained_pc_model.npz' not found.")
        st.info("Please run `python predictive_coding.py` first to train and save the model.")
        st.stop()
    return model

def preprocess_image(image):
    """Convert the uploaded image to the format the model expects."""
    # Convert to grayscale, resize to 28x28
    processed_image = image.convert('L').resize((28, 28))
    
    # Convert to numpy array
    image_array = np.array(processed_image)
    
    # Invert colors if necessary (model was trained on white digits on black background)
    # Check average color: if mostly white, invert it
    if np.mean(image_array) > 128:
        image_array = 255 - image_array

    # Normalize the image data just like in the training script
    # (pixels from 0-255 to -1 to 1)
    normalized_array = (image_array / 255.0 - 0.5) / 0.5
    
    # Flatten the array to a 784-element vector
    return normalized_array.flatten()

# --- UI Layout ---

st.title("🧠 Predictive Coding Network: MNIST Classifier")
st.write(
    "Upload an image of a handwritten digit (0-9) and the model will predict what it is. "
    "This model was trained from scratch using a biologically-inspired predictive coding algorithm."
)

# Load the trained model
model = load_model()

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # If a file is uploaded, open it as an image
    image = Image.open(uploaded_file)
    
    # Create two columns for a nice layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess the image for the model
    processed_image = preprocess_image(image)
    
    # Make a prediction using the model
    prediction = model.predict(processed_image)
    
    with col2:
        # Display the prediction in a large, clear format
        st.subheader("Model's Prediction:")
        st.markdown(f"<h1 style='text-align: center; color: #2E8B57;'>{prediction}</h1>", unsafe_allow_html=True)
        st.success(f"The model predicted the digit is a **{prediction}**.")
        st.balloons()

else:
    st.info("Please upload an image file to see a prediction.")

st.sidebar.header("About the Model")
st.sidebar.info(
    "This application uses a Predictive Coding Network (PCN) with the following architecture: "
    "**784 -> 256 -> 64 -> 10**. "
    "It was trained on the MNIST dataset for 3 epochs."
)