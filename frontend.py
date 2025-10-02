import streamlit as st
from PIL import Image
import torch
from backend import model, class_names, transform, device
import os

# ----------------- Setup -----------------
st.set_page_config(page_title="Face Shape Detector", page_icon="🖼️", layout="centered")

st.markdown("""
<style>
.header {
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    padding: 40px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
.header h1 {
    font-size: 42px;
    margin-bottom: 10px;
}
.header p {
    font-size: 18px;
}
.upload-box {
    margin-top: 20px;
    text-align: center;
}
.upload-btn {
    background-color: #6a11cb;
    color: white;
    padding: 10px 30px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
}
</style>
<div class="header">
    <h1>Face Shape Detector</h1>
    <p>Upload a photo of yourself and let the AI tell you your face shape.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- Image Upload -----------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Scan"):
        # Predict Face Shape
        img_tensor = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            pred_prob = probs.max().item()

        st.success(f"Prediction: {class_names[pred_label]} | Confidence: {pred_prob*100:.2f}%")
        st.image(image, caption=f"Predicted: {class_names[pred_label]}", use_container_width=True)

# ----------------- Face Shape Reference -----------------
st.markdown("## Face Shape Reference")
st.markdown("Here are the common face shapes and their descriptions:")

face_shapes = [
    {"name": "Round", "description": "Soft curves, cheeks and length roughly same width.", "path": r"C:\Users\thish\OneDrive\Pictures\Documents\DL\Face Detection\photos\round (9).jpg"},
    {"name": "Heart", "description": "Wider forehead & cheekbones, narrow chin.", "path": r"C:\Users\thish\OneDrive\Pictures\Documents\DL\Face Detection\photos\heart (1).jpg"},
    {"name": "Oval", "description": "Longer than wide, forehead slightly wider than chin.", "path": r"C:\Users\thish\OneDrive\Pictures\Documents\DL\Face Detection\photos\oval (1).jpg"},
    {"name": "Square", "description": "Strong jawline, forehead & jawline approx same width.", "path": r"C:\Users\thish\OneDrive\Pictures\Documents\DL\Face Detection\photos\square (3).jpg"},
    {"name": "Oblong", "description": "Longer than wide, forehead, cheekbones & jawline similar width.", "path": r"C:\Users\thish\OneDrive\Pictures\Documents\DL\Face Detection\photos\oblong (1000).jpg"},
]

thumbnail_size = (120, 120)
cols_per_row = 5

for i in range(0, len(face_shapes), cols_per_row):
    cols = st.columns(min(cols_per_row, len(face_shapes) - i))
    for j, shape in enumerate(face_shapes[i:i+cols_per_row]):
        with cols[j]:
            if os.path.exists(shape["path"]):
                img = Image.open(shape["path"]).convert("RGB")
                img.thumbnail(thumbnail_size)
                st.image(img, use_container_width=True)
            st.markdown(f"**{shape['name']}**")
            st.write(shape["description"])
