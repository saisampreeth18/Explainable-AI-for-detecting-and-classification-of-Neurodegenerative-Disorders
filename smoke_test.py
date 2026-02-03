import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Alzheimer MRI Analysis",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# SIDEBAR INFO
# ======================================================
with st.sidebar:
    st.markdown("## 🧠 Alzheimer MRI Analyzer")
    st.markdown("""
    **Purpose**  
    Academic AI system for analyzing brain MRI scans  

    **Approach**  
    Transfer Learning + Explainable AI (XAI)  

    **Explainability**  
    Grad-CAM and LIME  

    **Disclaimer**  
    This tool is for academic and research use only.  
    Not intended for clinical diagnosis.
    """)
    st.markdown("---")
    st.markdown("### 📊 Dataset Information")
    st.markdown("""
    **Dataset:** Alzheimer MRI (4 Classes)  
    **Classes:** Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented  
    **Note:** Subset used for training/evaluation.
    """)

# ======================================================
# CONFIG
# ======================================================
CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

MODEL_PATH = "alzheimer_model.h5"  # Your Kaggle .h5 file

# ======================================================
# LOAD MODEL - FIXED VERSION
# ======================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ {MODEL_PATH} not found in project folder!")
        st.info("💡 Download your Kaggle model and place 'alzheimer_model.h5' in the same folder as app.py")
        st.stop()
    
    st.info("🔄 Recreating model architecture and loading Kaggle weights...")
    
    try:
        # STEP 1: Recreate EXACT architecture (EfficientNetB0 + classifier head)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Add classifier head (standard for 4-class Alzheimer MRI)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.2)(x)
        predictions = Dense(4, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # STEP 2: Load WEIGHTS ONLY (bypasses shape error)
        model.load_weights(MODEL_PATH)
        
        # STEP 3: Compile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ Model loaded successfully via weights!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model weights: {str(e)}")
        st.info("💡 Make sure your Kaggle model has the same architecture (EfficientNetB0 + GAP + Dense)")
        st.stop()

# Load model at startup
try:
    model = load_model()
except:
    model = None

# ======================================================
# IMAGE PREPROCESSING
# ======================================================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0  # Normalize to [0,1]
    arr = np.expand_dims(arr, axis=0)
    return arr

# ======================================================
# GRAD-CAM
# ======================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=""):
    if not last_conv_layer_name:
        # Auto-detect last conv layer in EfficientNetB0
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if not last_conv_layer_name:
            raise ValueError("No convolutional layer found for Grad-CAM")
    
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = tf.reduce_max(predictions, axis=1)
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.cast(pooled_grads, tf.float32) * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ======================================================
# LIME EXPLAINER
# ======================================================
def lime_predict(images):
    return model.predict(np.array(images))

def generate_lime_explanation(img: Image.Image):
    explainer = lime_image.LimeImageExplainer()
    img_rgb = img.convert("RGB").resize((224,224))
    img_array = np.array(img_rgb)
    
    explanation = explainer.explain_instance(
        img_array,
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=300
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    lime_result = mark_boundaries(temp / 255.0, mask)
    return lime_result

# ======================================================
# STREAMLIT UI
# ======================================================
st.title("🧠 Explainable AI-Based Alzheimer's Disease Detection")
st.markdown("Upload a single **axial brain MRI image** (JPG/PNG) for analysis.")

col1, col2 = st.columns([3,1])
with col2:
    uploaded_file = st.file_uploader("Choose MRI image", type=["jpg","jpeg","png"])
    analyze_clicked = st.button("🔍 Analyze MRI", type="primary")

if model is None:
    st.error("❌ Model failed to load. Check file and architecture.")
elif uploaded_file is not None and analyze_clicked:
    with st.spinner("Processing MRI scan..."):
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="📁 Uploaded MRI", use_column_width=True)
        
        # Preprocess and predict
        processed_img = preprocess_image(img)
        preds = model.predict(processed_img, verbose=0)[0]
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))
        
        # Results
        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Stage", predicted_class)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        st.subheader("📈 Class Probabilities")
        probs_df = pd.DataFrame({
            'Class': CLASS_NAMES,
            'Probability': preds
        }).sort_values('Probability', ascending=False)
        st.dataframe(probs_df, use_container_width=True)
        
        # Progress bars
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, preds)):
            st.progress(prob)
            st.write(f"**{cls}:** {prob:.3f}")
        
        # Explainability tabs
        st.subheader("🔬 Explainable AI Visualizations")
        tab1, tab2 = st.tabs(["📈 Grad-CAM Heatmap", "🎯 LIME Explanation"])
        
        with tab1:
            try:
                heatmap = make_gradcam_heatmap(processed_img, model)
                heatmap = cv2.resize(heatmap, (224,224))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                img_array = np.array(img.convert("RGB").resize((224,224)))
                superimposed = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
                st.image(superimposed, caption="🔥 Grad-CAM: Red areas indicate important regions for diagnosis", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {str(e)}")
        
        with tab2:
            try:
                lime_result = generate_lime_explanation(img)
                st.image(lime_result, caption="🎯 LIME: Green regions most influence the prediction", use_column_width=True)
            except Exception as e:
                st.error(f"LIME failed: {str(e)}")
else:
    st.info("👆 Upload an MRI image and click **Analyze MRI** to get started!")
    st.markdown("---")
    st.markdown("""
    ### 🧠 **How it works:**
    1. **Upload** axial brain MRI (JPG/PNG)
    2. **AI predicts** dementia stage (4 classes)
    3. **Grad-CAM** shows which brain regions matter most
    4. **LIME** highlights pixels influencing the prediction
    """)
