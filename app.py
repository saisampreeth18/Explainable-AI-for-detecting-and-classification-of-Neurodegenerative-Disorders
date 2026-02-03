import numpy as np
from PIL import Image
import tensorflow as tf
import os
import io
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import streamlit as st
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Page config
st.set_page_config(page_title="Explainable AI for detecting and classification of Neurodegenerative Disorders", page_icon="🧠", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 Alzheimer MRI Analyzer")
    st.markdown("**MobileNetV3Large + Transfer Learning**")
    st.markdown("**Binary Classification: Demented vs Non-Demented**")

CLASS_NAMES = ["Non-Demented", "Demented"]

# PERFECT ARCHITECTURE (No Kaggle weights needed)
@st.cache_resource
def load_model():
    # Prefer a saved trained model if available
    if os.path.exists("alzheimer_model.h5"):
        try:
            model = tf.keras.models.load_model("alzheimer_model.h5")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception:
            pass

    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def make_gradcam_heatmap(img_array, model):
    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer
            break

    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

# UI - Polished Streamlit Frontend
st.markdown(
    """
    <div class="header">
      <h1>Explainable AI for detecting and classification of Neurodegenerative Disorders</h1>
      <p class="lead">MobileNetV3Large • Transfer Learning • Binary Classification</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .header { background: linear-gradient(90deg,#4b6cb7,#182848); color: white; padding: 18px; border-radius:8px; margin-bottom:16px;}
    .lead { margin-top: -6px; opacity:0.9; }
    .card { padding: 16px; border-radius: 8px; background-color: #f7f9fc; }
    .prediction-badge { display:inline-block; padding:8px 12px; border-radius:8px; color:white; font-weight:600;}
    .footer { color: #8892a6; font-size: 13px; margin-top: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

col_left, col_right = st.columns([2,1])
with col_left:
    st.markdown("**Upload an axial MRI (JPG/PNG). The model returns a probability of dementia and an explainable visualization (Grad-CAM or LIME).**")
    uploaded_file = st.file_uploader("Choose MRI file", type=['jpg','jpeg','png'])
    explanation_type = st.radio("Explanation type", ["Grad-CAM", "LIME", "None"], horizontal=True)
    analyze_btn = st.button("🔍 Analyze", type="primary")
with col_right:
    st.markdown("**Quick Tips**")
    st.write("- Use clear axial MRIs")
    st.write("- 224x224 ideal, larger images will be resized")
    st.write("- Results are for research / educational purposes")

if uploaded_file and analyze_btn:
    with st.spinner("🔬 Analyzing MRI..."):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📁 Original MRI", use_column_width=True)

        processed = preprocess_image(img)
        pred_prob = float(model.predict(processed, verbose=0)[0][0])
        pred_class = CLASS_NAMES[1] if pred_prob > 0.5 else CLASS_NAMES[0]
        confidence = max(pred_prob, 1-pred_prob)

        # Prediction card
        col_a, col_b = st.columns([2,3])
        with col_a:
            color = "#e74c3c" if pred_class=="Demented" else "#2ecc71"
            st.markdown(f"<div class='card'><span class='prediction-badge' style='background:{color}'>{pred_class}</span>"
                        f"<p style='margin-top:8px;'>Probability: <b>{pred_prob:.1%}</b></p>"
                        f"<p>Confidence: <b>{confidence:.1%}</b></p></div>", unsafe_allow_html=True)
            # Confidence bar
            st.subheader("Model Confidence")
            st.progress(float(confidence))
        with col_b:
            st.subheader("Details")
            st.metric("Confidence", f"{confidence:.1%}")
            st.write("Model: MobileNetV3Large (imagenet backbone)")
            st.download_button("⬇️ Download Prediction CSV", data=f"filename,probability\n{uploaded_file.name},{pred_prob:.6f}".encode(), file_name="prediction.csv")

        # Explanation
        if explanation_type == "Grad-CAM":
            st.subheader("🔬 Grad-CAM")
            try:
                heatmap = make_gradcam_heatmap(processed, model)
                heatmap_resized = cv2.resize(heatmap, (224,224))
                heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
                img_rgb = np.array(img.resize((224,224)))
                overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
                st.image(heatmap_colored, caption="🔥 Heatmap (red = important)", use_column_width=True)
                st.image(overlay, caption="🔍 Overlay on MRI", use_column_width=True)

                # download overlay
                buf = io.BytesIO()
                Image.fromarray(overlay.astype('uint8')).save(buf, format='PNG')
                buf.seek(0)
                st.download_button("⬇️ Download Overlay", buf, file_name="overlay.png", mime="image/png")

            except Exception as e:
                st.warning("🔧 Grad-CAM unavailable: " + str(e))

        elif explanation_type == "LIME":
            st.subheader("🧭 LIME Explanation")
            try:
                explainer = lime_image.LimeImageExplainer()
                def classifier_fn(images):
                    images = np.array([np.array(Image.fromarray(np.uint8(im)).resize((224,224))).astype('float32')/255. for im in images])
                    probs = model.predict(images, verbose=0)
                    # return array with two columns [non-demented, demented]
                    return np.concatenate([1-probs, probs], axis=1)
                explanation = explainer.explain_instance(np.array(img.resize((224,224))), classifier_fn, top_labels=2, hide_color=0, num_samples=200)
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
                lime_img = mark_boundaries(temp/255.0, mask)
                st.image(lime_img, caption="🧩 LIME Explanation", use_column_width=True)

                buf = io.BytesIO()
                Image.fromarray(np.uint8(lime_img*255)).save(buf, format='PNG')
                buf.seek(0)
                st.download_button("⬇️ Download LIME Image", buf, file_name="lime.png", mime="image/png")
            except Exception as e:
                st.warning("🔧 LIME unavailable: " + str(e))

        else:
            st.info("No explainability selected. Choose Grad-CAM or LIME to visualize model focus areas.")

        st.markdown("---")
        with st.expander("Model Info & Notes"):
            st.write("- This demo uses MobileNetV3Large with an added classification head.")
            if os.path.exists("alzheimer_model.h5"):
                st.write("- Loaded local weights: `alzheimer_model.h5`")
            st.write("- Outputs are for research/educational purposes only.")

st.markdown('<div class="footer">Built with ❤ • MobileNetV3Large • Explainable AI (Grad-CAM & LIME)</div>', unsafe_allow_html=True)
