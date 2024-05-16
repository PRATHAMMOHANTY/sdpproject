import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from skimage.transform import resize
import tensorflow as tf

N = 512

# Function to generate masked image
def generate_masked_image(input_image, model):
    # Convert the image to numpy array
    img = np.array(input_image)
    
    # Resize the image
    img = resize(img, (N, N,3), mode="constant",preserve_range=True).astype(np.float32)
    
    # Normalize pixel values
    # img = img / 255.0
    
    # Expand dimensions to match the model input shape
    img = np.expand_dims(img, axis=0)
    
    # Generate the masked image
    pred_mask = model(img)
    masked_image = np.squeeze(pred_mask)
    
    # Create a figure and plot the masked image
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(masked_image)
    plt.show()
    
    # Convert the figure to a PIL Image
    fig.canvas.draw()
    masked_image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    
    return masked_image

# Streamlit app
st.title("Image Masking")

# Model loading
model_path = r"SDPF/vgg"
model = tf.saved_model.load(model_path)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    input_image = Image.open(uploaded_file)

    # Generate the masked image
    masked_image = generate_masked_image(input_image, model)

    # Display the original and masked images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_column_width=True)
    with col2:
        st.subheader("Masked Image")
        st.image(masked_image, use_column_width=True)
