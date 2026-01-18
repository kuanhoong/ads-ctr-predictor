
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model_2d_cnn import CNN2D

# Config
MODEL_PATH = "model_2d.pth"
DEVICE = torch.device("cpu") # CPU for inference usually fine

@st.cache_resource
def load_model():
    model = CNN2D()
    # Handle state dict load
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        st.error(f"Model file {MODEL_PATH} not found. Please run training first.")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    """
    Input: PIL Image
    Output: Tensor (1, 3, 144, 144)
    """
    # Transform pipeline matching training
    transform = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.ToTensor(),
        # Normalize? Training used 0-1 (ToTensor only). 
        # Actually dummy data was 0-255 -> ToTensor -> 0-1.
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def main():
    st.set_page_config(page_title="Ad CTR Predictor", layout="wide")
    
    st.title("🏆 Ad Design Optimizer")
    st.markdown("Upload 3 ad variations to predict their Click-Through Rate (CTR) and find the winner.")
    
    # Load Model
    model = load_model()
    if model is None:
        return

    # Uploads
    col1, col2, col3 = st.columns(3)
    
    uploaded_files = []
    
    with col1:
        st.subheader("Design A")
        file1 = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg"], key="imp1")
        if file1: uploaded_files.append(("Design A", file1))
        
    with col2:
        st.subheader("Design B")
        file2 = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg"], key="imp2")
        if file2: uploaded_files.append(("Design B", file2))
        
    with col3:
        st.subheader("Design C")
        file3 = st.file_uploader("Upload Image C", type=["png", "jpg", "jpeg"], key="imp3")
        if file3: uploaded_files.append(("Design C", file3))
        
    if len(uploaded_files) == 3:
        st.divider()
        st.subheader("Analysis Results")
        
        results = []
        
        # Grid for display
        res_col1, res_col2, res_col3 = st.columns(3)
        cols = [res_col1, res_col2, res_col3]
        
        for idx, (name, file) in enumerate(uploaded_files):
            # Load and Display
            image = Image.open(file).convert("RGB")
            cols[idx].image(image, caption=name, use_container_width=True)
            
            # Predict
            img_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(img_tensor)
                ctr_score = output.item()
                
            results.append((name, ctr_score))
            cols[idx].metric(label="Predicted CTR", value=f"{ctr_score:.4f}")
            
        # Recommend
        st.divider()
        best_design, best_score = max(results, key=lambda x: x[1])
        st.success(f"🌟 Recommendation: **{best_design}** is predicted to perform best with a CTR of **{best_score:.4f}**")
        
        # Plot comparison if needed (bar chart)
        chart_data = {
            "Design": [r[0] for r in results],
            "CTR": [r[1] for r in results]
        }
        st.bar_chart(data=chart_data, x="Design", y="CTR")
        
    else:
        st.info("Please upload all 3 designs to compare.")

if __name__ == "__main__":
    main()
