import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Pneumonia Detection",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def get_model():
    try:
        return load_model("densenet121_pneumonia.keras")
    except:
        st.error("Model file not found. Please ensure densenet121_pneumonia.keras is in the app directory.")
        return None

model = get_model()

# =========================
# IMAGE PREPROCESSING
# =========================

IMG_SIZE = (224, 224)

def preprocess_image(img):

    img = img.resize(IMG_SIZE)

    img_array = np.array(img)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# GRADCAM
# =========================

def generate_gradcam(model, img_array):

    last_conv_layer_name = "conv5_block16_concat"

    last_conv_layer = model.get_layer(
        last_conv_layer_name
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / np.max(heatmap)

    return heatmap

# =========================
# TABS
# =========================

tab1, tab2 = st.tabs(
    [
        "Pneumonia Detection",
        "Project Information"
    ]
)

# ====================================================
# TAB 1
# ====================================================

with tab1:

    st.title(
        "Chest X-Ray Pneumonia Detection"
    )

    st.write(
        "Upload a chest X-ray image for prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload X-Ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded X-Ray",
            width=350
        )

        if model is None:
            st.error("Cannot make predictions - model not loaded")
        else:
            img_array = preprocess_image(
                image
            )

            prediction = model.predict(
                img_array,
                verbose=0
            )[0][0]

            if prediction > 0.5:

                label = "PNEUMONIA"
                confidence = prediction

            else:

                label = "NORMAL"
                confidence = 1 - prediction

            st.subheader(
                f"Prediction: {label}"
            )

            st.write(
                f"Confidence: {confidence*100:.2f}%"
            )

            st.progress(
                float(confidence)
            )

        # ------------------
        # GRADCAM
        # ------------------

        if model is not None:
            try:

                heatmap = generate_gradcam(
                    model,
                    img_array
                )

                original = np.array(image)

                heatmap_resized = cv2.resize(
                    heatmap,
                    (
                        original.shape[1],
                        original.shape[0]
                    )
                )

                heatmap_colored = cv2.applyColorMap(
                    np.uint8(
                        255 * heatmap_resized
                    ),
                    cv2.COLORMAP_JET
                )

                heatmap_colored = cv2.cvtColor(
                    heatmap_colored,
                    cv2.COLOR_BGR2RGB
                )

                overlay = cv2.addWeighted(
                    original,
                    0.6,
                    heatmap_colored,
                    0.4,
                    0
                )

                st.subheader(
                    "Model Attention"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        heatmap_colored,
                        caption="Grad-CAM Heatmap"
                    )

                with col2:
                    st.image(
                        overlay,
                        caption="Overlay"
                    )

            except Exception as e:

                st.warning(
                    f"Grad-CAM error: {e}"
                )

# ====================================================
# TAB 2
# ====================================================

with tab2:

    st.title(
        "Project Overview"
    )

    st.header(
        "Objective"
    )

    st.write("""
    Detect pneumonia from chest X-ray images
    using Deep Learning and Transfer Learning.
    """)

    st.header(
        "Dataset"
    )

    st.write("""
    Kaggle Chest X-Ray Images (Pneumonia)

    Classes:
    - NORMAL
    - PNEUMONIA

    Approximately 5,800 images.
    """)

    st.header(
        "Models Evaluated"
    )

    st.write("""
    1. CNN from Scratch
    2. VGG16 Transfer Learning
    3. DenseNet121 Transfer Learning
    """)

    st.header(
        "Why DenseNet121?"
    )

    st.write("""
    DenseNet connects each layer to every
    subsequent layer, improving feature reuse
    and gradient flow.

    Advantages:
    - Better feature extraction
    - Reduced vanishing gradients
    - Strong performance on medical imaging
    - Higher accuracy than CNN and VGG16
    """)

    st.header(
        "Transfer Learning"
    )

    st.write("""
    DenseNet121 was pretrained on ImageNet.

    Phase 1:
    - Backbone frozen
    - Only classifier trained

    Phase 2:
    - Entire network unfrozen
    - Fine-tuned using a lower learning rate
    """)

    st.header(
        "Evaluation Metrics"
    )

    st.write("""
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - AUC-ROC
    """)

    st.header(
        "Best DenseNet Results"
    )

    st.write("""
    Accuracy : 92.31%

    Precision : 90.33%

    Recall : 98.21%

    F1 Score : 94.10%

    AUC-ROC : 97.66%
    """)

    st.header(
        "Explainability"
    )

    st.write("""
    Grad-CAM is used to visualize
    which regions of the chest X-ray
    contributed most to the model's decision.

    This improves transparency and
    interpretability in medical AI systems.
    """)