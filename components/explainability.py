import streamlit as st
from pathlib import Path
from PIL import Image
from utils.inference import get_history
from utils.gradcam_pp import generate_gradcam


@st.cache_data
def load_and_generate_gradcam(image_path: str, case_id: str):
    """
    Load image and generate Grad-CAM++ visualization with caching.
    
    Args:
        image_path: Path to the original image
        case_id: Case ID for saving the Grad-CAM output
    
    Returns:
        tuple: (original_image, gradcam_overlay)
    """
    # Check if Grad-CAM already exists on disk
    gradcam_path = Path(f"{case_id}_gradcam.png")
    
    # Load original image
    original_image = Image.open(image_path).convert("RGB")
    
    # Load or generate Grad-CAM
    if gradcam_path.exists():
        gradcam_overlay = Image.open(gradcam_path)
    else:
        gradcam_overlay = generate_gradcam(original_image, case_id)
    
    return original_image, gradcam_overlay


def render_explainability():
    st.markdown("## 🧠 Grad-CAM++ Explainability")
    
    # Get history
    history = get_history()
    
    if not history or len(history) == 0:
        st.warning("⚠️ No cases available. Please upload an image first in the 'Upload & Scan' page.")
        return
    
    # Get most recent case
    most_recent = history[-1]
    case_id = most_recent["case_id"]
    image_path = most_recent["image_path"]
    label = most_recent["label"]
    
    # Check if image exists
    if not Path(image_path).exists():
        st.error(f"❌ Image file not found: {image_path}")
        st.info("The image may have been moved or deleted. Please upload a new image.")
        return
    
    # Display case info
    st.markdown(f"**Analyzing Case:** `{case_id}`")
    st.markdown(f"**Verdict:** {'🔴 FORGED' if label == 'forged' else '🟢 AUTHENTIC' if label == 'authentic' else '⚠️ ERROR'}")
    st.markdown("---")
    
    # Load and generate Grad-CAM
    try:
        with st.spinner("🔥 Generating Grad-CAM++ heatmap..."):
            original_image, gradcam_overlay = load_and_generate_gradcam(image_path, case_id)
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(original_image, use_container_width=True)
        
        with col2:
            st.markdown("### 🔥 Grad-CAM++ Heatmap")
            st.image(gradcam_overlay, use_container_width=True)
        
        # Caption
        st.info(
            "💡 **Grad-CAM++ Visualization**: The heatmap highlights regions that contribute most to the model's "
            "forgery prediction. Red/yellow areas indicate high activation (regions of interest), while "
            "blue/green areas indicate lower activation."
        )
        
        # Additional explanation
        with st.expander("ℹ️ How Grad-CAM++ Works"):
            st.markdown("""
            **Grad-CAM++** (Gradient-weighted Class Activation Mapping) is an explainability technique that:
            
            1. **Captures Gradients**: Records how the model's output changes with respect to feature maps
            2. **Computes Importance Weights**: Uses gradient information to determine which regions are most important
            3. **Generates Heatmap**: Creates a visual overlay showing areas the model focuses on
            
            This helps understand:
            - Which parts of the image the model considers suspicious
            - Whether the model is focusing on relevant features
            - Potential biases or unexpected behavior in the model
            
            **Target Layer**: `model.seg_head.net[-1]` (final decoder convolution layer)
            """)
    
    except Exception as e:
        st.error(f"❌ Error generating Grad-CAM++: {str(e)}")
        st.exception(e)
