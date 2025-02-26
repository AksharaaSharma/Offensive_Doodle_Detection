import streamlit as st
import google.generativeai as genai
import torch
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import base64
import io
import os

# Configure Gemini API
genai.configure(api_key="AIzaSyCW6X3nK9yF4Q-XNN-2nl3j3wYfoCv32zc")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def load_clip_model():
    # Path to your fine-tuned model
    model_path = r'fine_tuned_clip_model'
    
    try:
        # Load the model configuration manually
        from transformers import CLIPModel, CLIPProcessor, AutoConfig
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_path)
        
        # Load the model weights
        model = CLIPModel.from_pretrained(
            model_path, 
            local_files_only=True,
            config=config
        )
        
        # For processor, we'll use the default pretrained processor
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # Set to evaluation mode
        model.eval()
        
        return model, processor
    except Exception as e:
        st.error(f"Error loading CLIP model: {e}")
        return None, None

# Preprocessing function for CLIP
def preprocess_for_clip(image, processor):
    """Preprocess image for CLIP model"""
    # Prepare both image and text for CLIP model
    reasoning_offensive = "This doodle is offensive because it depicts harmful content."
    reasoning_non_offensive = "This doodle is non-offensive because it depicts harmless content."
    
    # Preprocess both offensive and non-offensive scenarios
    inputs_offensive = processor(
        text=[reasoning_offensive], 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    
    inputs_non_offensive = processor(
        text=[reasoning_non_offensive], 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    
    return inputs_offensive, inputs_non_offensive

# CLIP Model Classification
def classify_with_clip(image, clip_model, processor):
    """Classify image using fine-tuned CLIP model"""
    if clip_model is None or processor is None:
        st.warning("CLIP model not loaded")
        return None
    
    try:
        # Preprocess for both offensive and non-offensive scenarios
        inputs_offensive, inputs_non_offensive = preprocess_for_clip(image, processor)
        
        # Get similarities for both scenarios
        with torch.no_grad():
            outputs_offensive = clip_model(**inputs_offensive)
            outputs_non_offensive = clip_model(**inputs_non_offensive)
        
        # Extract logits
        logits_offensive = outputs_offensive.logits_per_image[0][0].item()
        logits_non_offensive = outputs_non_offensive.logits_per_image[0][0].item()
        
        # Determine classification based on logits
        return "OFFENSIVE" if logits_offensive > logits_non_offensive else "NON OFFENSIVE"
    
    except Exception as e:
        st.error(f"CLIP Classification Error: {e}")
        return None

# Gemini Content Analysis
def analyze_content(image_bytes):
    try:
        # Detailed prompt for Gemini
        prompt = [
            "Carefully analyze this hand-drawn image. Determine if the drawing contains any offensive content.",
            "Provide a detailed analysis:",
            "1. Offensive Classification: (Yes/No)",
            "2. Confidence Level: X%",
            "3. Detailed Explanation",
            "4. Specific Offensive Elements (if any)"
        ]
        
        # Generate response using Gemini
        response = gemini_model.generate_content(
            prompt + [{'mime_type': 'image/png', 'data': image_bytes}],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7,
                top_p=1.0
            )
        )
        return response.text
    except Exception as e:
        st.error(f"Error analyzing content: {e}")
        return None

# Main Streamlit Application
def main():
    # Load CLIP Model
    clip_model, clip_processor = load_clip_model()
    
    # Styling with updated title CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    h1 {
        font-family: 'Inter', sans-serif;
        color: #5DBFD5;  /* Updated to light blue color */
        text-align: center;
        font-size: 5.5rem;
        font-weight: 900;
        margin-bottom: 30px;
        letter-spacing: -2px;
        text-shadow: 
            0 0 20px rgba(93, 191, 213, 0.5),
            0 0 30px rgba(93, 191, 213, 0.4),
            0 0 40px rgba(93, 191, 213, 0.3);
        animation: titleGlow 2s ease-in-out infinite alternate;
        white-space: nowrap;
    }

    @keyframes titleGlow {
        0% { 
            text-shadow: 
                0 0 20px rgba(93, 191, 213, 0.3),
                0 0 30px rgba(93, 191, 213, 0.2),
                0 0 40px rgba(93, 191, 213, 0.1);
            color: #5DBFD5;
        }
        100% { 
            text-shadow: 
                0 0 30px rgba(93, 191, 213, 0.7),
                0 0 40px rgba(93, 191, 213, 0.6),
                0 0 50px rgba(93, 191, 213, 0.5);
            color: #5DBFD5;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Background Image
    background_image_path = r"Thales_Offensive_Doodle.jpg"
    with open(background_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar and Canvas setup
    with st.sidebar:
        st.markdown("<h2 class='sidebar-text'>🎨 Drawing Toolkit</h2>", unsafe_allow_html=True)
        
        uploaded_doodle = st.file_uploader(
            "Upload Your Doodle", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an existing doodle to analyze"
        )
        
        stroke_width = st.slider(
            "Brush Thickness", 
            min_value=1, 
            max_value=20,
            value=5,
            help="Adjust your brush size 🖌️"
        )
        
        stroke_color = st.color_picker(
            "Line Color", 
            value="#5DBFD5",
            help="Choose your drawing color ✨"
        )
        
        drawing_mode = st.radio(
            "Drawing Style", 
            ["Freedraw 🖍️", "Line ➖", "Rectangle ◼️", "Circle ⭕", "Eraser 🧽"],
            help="Pick your drawing mode"
        )

    # Glowy Title
    st.markdown("<h1>Doodle Detector 🕵️‍♀️</h1>", unsafe_allow_html=True)
    
    # Mapping drawing mode to st_canvas mode
    mode_map = {
        "Freedraw 🖍️": "freedraw",
        "Line ➖": "line",
        "Rectangle ◼️": "rect",
        "Circle ⭕": "circle",
        "Eraser 🧽": "freedraw"
    }
    
    # Determine stroke color
    active_stroke_color = stroke_color if "Eraser" not in drawing_mode else "#FFFFFF"
    
    # Create drawing canvas or show uploaded doodle
    if uploaded_doodle:
        # Display uploaded doodle
        doodle_image = Image.open(uploaded_doodle)
        st.image(doodle_image, caption="Your Uploaded Doodle", use_column_width=True)
        
        # Convert to bytes for analysis
        img_byte_arr = io.BytesIO()
        doodle_image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
    else:
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
            stroke_width=stroke_width,
            stroke_color=active_stroke_color,
            background_color="#FFFFFF",  # White background
            height=500,
            width=700,
            drawing_mode=mode_map[drawing_mode],
            key="canvas"
        )
    
    # Analyze Content Button
    if st.button("Detect Offensiveness 🕵️"):
        # Use either uploaded image or canvas drawing
        if uploaded_doodle:
            img = doodle_image
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()
        elif canvas_result.image_data is not None:
            img = Image.fromarray(np.uint8(canvas_result.image_data))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()
        else:
            st.warning("Draw something or upload a doodle before our detective can investigate! 🎨")
            return
        
        # CLIP Model Classification
        if clip_model is not None:
            clip_classification = classify_with_clip(img, clip_model, clip_processor)
        
        # Gemini Analysis
        gemini_analysis = analyze_content(img_bytes)
        
        # Display Results
        if gemini_analysis:
            # Determine final classification
            if clip_classification and clip_classification in gemini_analysis:
                final_classification = clip_classification
            else:
                # Extract Gemini's classification
                lines = gemini_analysis.split("\n")
                gemini_classification = next((line.split(":")[1].strip() for line in lines if "Offensive Classification" in line), None)
                final_classification = gemini_classification or "UNDETERMINED"
            
            # Display results 
            st.markdown(f"<h1 style='text-align: center; color: {'#2e5984' if final_classification == 'NON OFFENSIVE' else '#FF0000'};'>Content Analysis Complete</h1>", unsafe_allow_html=True)
            st.write(gemini_analysis)
        else:
            st.error("Oops! Our detective couldn't crack the case 🕵️‍♀️🤷‍♀️")

if __name__ == "__main__":
    main()
