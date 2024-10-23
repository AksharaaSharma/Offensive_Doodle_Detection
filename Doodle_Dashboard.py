import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
import base64
import pandas as pd
from datetime import datetime  # Import for date and time

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Define the necessary transformations for the input image (match with your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your pretrained model
model_path = r"Resnet.pth"  # Update with your saved model path

# Define the model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Binary classification

# Load the model weights
pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
# Filter out `fc` layers
pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc.')}
model.load_state_dict(pretrained_dict, strict=False)
model.eval()  # Set the model to evaluation mode

# Define the folder where all new doodles will be saved
new_doodle_folder = r"New Doodle Images"  # Update with your desired path

# Ensure the folder exists
if not os.path.exists(new_doodle_folder):
    os.makedirs(new_doodle_folder)

# CSV file to save labels
csv_file_path = os.path.join(new_doodle_folder, r"New Doodle.csv")

# Create CSV if it doesn't exist
if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=["Image Name", "Label"])
    df.to_csv(csv_file_path, index=False)

# Function to save doodle and label
def save_doodle_with_label(image_data, prediction):
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Define the image name (with date and time)
    img_name = f"doodle_{current_time}.png"
    img_path = os.path.join(new_doodle_folder, img_name)
    
    # Convert the canvas image data to a PIL image
    img_pil = Image.fromarray(np.uint8(image_data)).convert('RGB')
    
    # Save the image
    img_pil.save(img_path)

    # Save label to CSV
    label = "Offensive" if prediction >= 0.5 else "Non-Offensive"
    df = pd.read_csv(csv_file_path)
    new_entry = pd.DataFrame([{"Image Name": img_name, "Label": label}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file_path, index=False)
    return img_path

# Title of the app with increased size and Comic Sans font
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        font-size: 50px;
    }
    </style>
    <h1 class="title">Offensive Doodle Detector</h1>
    """, 
    unsafe_allow_html=True
)

# Convert the background image to base64
background_image_path = r"wp9650652-desktop-doodle-4k-wallpapers.jpg"  # Update with your background image path
background_image_base64 = image_to_base64(background_image_path)

# CSS for placing the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{background_image_base64}");
        background-size: cover;
    }}
    .stSidebar {{
        background-color: rgba(0, 0, 0, 0.4);
            }}
    .sidebar-title {{
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: white;
        font-size: 30px;
        text-align: center;
    }}
    .sidebar-select {{
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: white;
        font-size: 20px;
    }}
    .result {{
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: white;
        font-size: 24px;
        text-align: center;
        display: inline-block;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Toggle for drawing or erasing mode
st.sidebar.markdown("<h2 class='sidebar-title'>Canvas Controls</h2>", unsafe_allow_html=True)
drawing_mode = st.sidebar.selectbox("Select mode:", ["freedraw", "erase"])

# Stroke width slider for drawing and erasing
if drawing_mode == "freedraw":
    stroke_width = st.sidebar.slider("Stroke Width", min_value=1, max_value=30, value=5)
else:  # Eraser mode
    stroke_width = st.sidebar.slider("Eraser Width", min_value=10, max_value=50, value=25)

# Create columns for centering
col1, col2, col3 = st.columns([1, 5, 1])

with col2:
    # Create a white drawing canvas with different stroke colors and thickness
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#000000" if drawing_mode == "freedraw" else "#FFFFFF",  # Use white for eraser
        background_color="#FFFFFF",  # Keep the canvas background white
        width=500,
        height=500,
        drawing_mode="freedraw",  # Both drawing and erasing use the "freedraw" mode
        key="canvas",
    )

    if st.button("Submit Doodle", key="submit_button", help="Click to submit your doodle for classification"):
        if canvas_result.image_data is not None:
            # Convert the canvas image data to a PIL image
            img = Image.fromarray(np.uint8(canvas_result.image_data)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            # Inference
            with torch.no_grad():
                output = model(img_tensor).squeeze(0)
                prediction = torch.sigmoid(output).item()

            # Determine if the doodle is offensive or not
            result_text = "The doodle is OFFENSIVE." if prediction >= 0.5 else "The doodle is NON-OFFENSIVE."
            st.markdown(f"<h2 class='result'>{result_text}</h2>", unsafe_allow_html=True)

            # Save the doodle and label it
            save_doodle_with_label(canvas_result.image_data, prediction)
        else:
            st.write("Please draw a doodle first!")

# Upload Doodle functionality
st.sidebar.markdown("<h2 class='sidebar-title'>Upload Doodle</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your doodle :", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    uploaded_img = Image.open(uploaded_file).convert('RGB')

    col_center = st.columns(3)[1]  # Get the middle column for centering
    with col_center:
        st.image(uploaded_img, caption="Uploaded Doodle", use_column_width=False, width=300)

    if st.button("Submit Uploaded Doodle", help="Click to submit your uploaded doodle for classification"):
        img = transform(uploaded_img).unsqueeze(0)

        with torch.no_grad():
            output = model(img).squeeze(0)
            prediction = torch.sigmoid(output).item()

        result_text = "The uploaded doodle is OFFENSIVE." if prediction >= 0.5 else "The uploaded doodle is NON-OFFENSIVE."
        st.markdown(f"<h2 class='result'>{result_text}</h2>", unsafe_allow_html=True)
