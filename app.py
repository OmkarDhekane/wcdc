import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
import warnings
import absl.logging
from utils_gradcam import make_gradcam_heatmap, overlay_heatmap
from dotenv import load_dotenv
import traceback
from huggingface_hub import hf_hub_download

load_dotenv()

# Suppress logs
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==== CONFIG ====
MODEL_NAME = "aug_alb_best_model_DenseNet121_FT.h5"
HF_MODEL_REPO = "OmkarDhekane/wcdc_densenet121_AUGFT"
TOKEN = os.getenv("HF_TOKEN")

LABELS = [
    "healthy",
    "leaf_rust",
    "powdery_mildew",
    "seedlings",
    "septoria",
    "stem_rust",
    "yellow_rust",
]
IMG_SIZE = (240, 240)
BACKBONE_NAME = "densenet121"  # from model.summary()

# ==== LOAD MODEL ====
@st.cache_resource
def load_model(hf_repo: str, model_name: str, token: str | None):
    try:
        print(f"Attempting to download {model_name} from {hf_repo}...")
        h5_model_path = hf_hub_download(
            repo_id=hf_repo,
            filename=model_name,
            token=token,
        )
        model = tf.keras.models.load_model(h5_model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"🚫 Failed to load model: {e}")
        st.code(traceback.format_exc())
        return None


model = load_model(HF_MODEL_REPO, MODEL_NAME, TOKEN)
# print(model.summary())
print("Model loaded successfully!")

# ==== SIDEBAR INFO ====
st.sidebar.title("📦 Model Info")
st.sidebar.markdown(f"**Model Name:** `{MODEL_NAME}`")
st.sidebar.markdown("**Supported Classes:**")
for label in LABELS:
    st.sidebar.markdown(f"- {label}")

threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("---")
st.sidebar.info("Predictions with confidence > threshold are shown as positive.")

# ==== MAIN APP ====
st.title("🌾 Wheat Crop Disease Detector")
st.markdown(
    "Upload an image of a wheat leaf and predict possible diseases using a deep learning model."
)

uploaded = st.file_uploader("📤 Choose an image", type=["jpg", "png", "jpeg"])

if uploaded is not None and model is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict", use_container_width=True):
        try:
            # --- Preprocess image ---
            input_img = np.expand_dims(
                np.array(image.resize(IMG_SIZE)) / 255.0,
                axis=0,
            ).astype("float32")  # (1, 240, 240, 3)

            # --- Predict ---
            pred = model.predict(input_img)[0]  # shape: (7,)

            pred_df = (
                pd.DataFrame(
                    {
                        "Label": LABELS,
                        "Confidence (%)": (pred * 100).round(2),
                        "Predicted": [1 if p > threshold else 0 for p in pred],
                    }
                )
                .sort_values(by="Confidence (%)", ascending=False)
                .reset_index(drop=True)
            )

            st.subheader("📊 Prediction Results")
            st.dataframe(
                pred_df,#.drop(columns=["Predicted"]),
                use_container_width=True,
            )

            # --- Grad-CAM for detected classes ---
            detected = pred_df[pred_df["Predicted"] == 1]

            if not detected.empty:
                st.markdown("### 🧾 Detected Diseases with Grad-CAM")

                for label in detected["Label"].values:
                    class_idx = LABELS.index(label)

                    # Compute Grad-CAM heatmap for this class
                    heatmap = make_gradcam_heatmap(
                        img_array=input_img,
                        model=model,
                        backbone_name=BACKBONE_NAME,
                        class_index=class_idx,
                    )

                    gradcam_img = overlay_heatmap(heatmap, image)

                    st.markdown(f"#### 🔍 {label}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            image,
                            caption="Uploaded Leaf",
                            use_container_width=True,
                        )
                    with col2:
                        st.image(
                            gradcam_img,
                            caption="Grad-CAM Visualization",
                            use_container_width=True,
                        )
            else:
                # If nothing passes threshold, still show Grad-CAM for top-1 class
                top_idx = int(np.argmax(pred))
                top_label = LABELS[top_idx]

                st.info(
                    f"✅ No class passed the threshold {threshold:.2f}. "
                    f"Showing Grad-CAM for top prediction: **{top_label}**."
                )

                heatmap = make_gradcam_heatmap(
                    img_array=input_img,
                    model=model,
                    backbone_name=BACKBONE_NAME,
                    class_index=top_idx,
                )
                gradcam_img = overlay_heatmap(heatmap, image)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        image,
                        caption="Uploaded Leaf",
                        use_container_width=True,
                    )
                with col2:
                    st.image(
                        gradcam_img,
                        caption=f"Grad-CAM for {top_label}",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.code(traceback.format_exc())
