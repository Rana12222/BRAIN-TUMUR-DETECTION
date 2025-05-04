import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io

# --- Define the classes ---
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# --- Load YOLO model ---
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"🚫 Failed to load model: {e}")
        return None

# --- Detect and visualize results ---
def detect_and_plot(image, model):
    results = model.predict(image)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    
    detected_class = None

    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = detection.cls[0].cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f"{classes[int(cls)]} {conf:.2f}",
                 color='white', fontsize=12, backgroundcolor='red')

        if conf > 0.5:
            detected_class = classes[int(cls)]

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf, detected_class

# --- Streamlit App Layout ---
st.set_page_config(page_title="🧠 Brain Tumor Detector", layout="centered")

st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#FF4B4B;">🧠 Brain Tumor Detector</h1>
        <p style="font-size:18px;">Upload a brain MRI image to detect possible tumors using YOLOv8.</p>
    </div>
    <hr style="margin-top:-10px;">
""", unsafe_allow_html=True)

# --- Upload Section ---
with st.container():
    st.subheader("📤 Upload MRI Image")
    uploaded_image = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])

# --- Process Image ---
if uploaded_image:
    with st.expander("📸 Preview Uploaded Image", expanded=True):
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    image_np = np.array(image)
    model_path = "model.pt"  # <-- Update your model path here
    model = load_model(model_path)

    if model:
        with st.spinner("🔍 Detecting tumor..."):
            result_img, tumor_type = detect_and_plot(image_np, model)

        st.success("✅ Detection Completed!")

        st.image(result_img, caption="🧪 Detection Result", use_column_width=True)

        if tumor_type:
            st.markdown(f"""
            <div style="text-align:center; margin-top:20px;">
                <h3 style="color:green;">🧬 Detected Tumor Type: <span style="color:#FF4B4B;">{tumor_type}</span></h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No tumor detected with high confidence.")
else:
    st.info("Please upload a brain MRI image to begin analysis.")
