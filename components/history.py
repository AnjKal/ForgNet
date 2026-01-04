import streamlit as st
from utils import inference
from pathlib import Path
from PIL import Image
import numpy as np

def render_history():
    """
    Render the history page showing all past predictions.
    """
    history = inference.get_history()
    
    if not history:
        st.markdown('<div class="glass-panel" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
        st.markdown("### 📭 No History Yet")
        st.markdown("Upload and scan images to see them here.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown(f"### 📊 Total Cases: {len(history)}")
    st.markdown("---")
    
    # Display in reverse chronological order (newest first)
    for idx, record in enumerate(reversed(history)):
        case_id = record.get("case_id", "Unknown")
        label = record.get("label", "unknown")
        timestamp = record.get("timestamp", "N/A")
        image_path = record.get("image_path", "")
        annotation = record.get("annotation", "")
        
        # Determine status color
        if label == "authentic":
            status_color = "#00ff88"
            status_icon = "✓"
        elif label == "fake":
            status_color = "#ff4444"
            status_icon = "⚠"
        else:
            status_color = "#888888"
            status_icon = "?"
        
        # Create expandable card for each case
        with st.expander(f"**{status_icon} Case {case_id}** - {label.upper()}", expanded=(idx == 0)):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image if it exists
                if image_path and Path(image_path).exists():
                    try:
                        img = Image.open(image_path)
                        st.image(img, caption=f"Case {case_id}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load image: {e}")
                else:
                    st.warning("Image not found")
            
            with col2:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown(f"**Case ID:** `{case_id}`")
                st.markdown(f"**Verdict:** <span style='color: {status_color}; font-weight: bold;'>{label.upper()}</span>", unsafe_allow_html=True)
                st.markdown(f"**Timestamp:** {timestamp}")
                
                if annotation != "authentic" and annotation:
                    st.markdown(f"**RLE Annotation:** `{annotation[:100]}...`" if len(annotation) > 100 else f"**RLE Annotation:** `{annotation}`")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display probability map if available
                if image_path and Path(image_path).exists() and label == "fake":
                    # Try to load and display the mask
                    # For now, we'll re-run inference to get the mask (not ideal but works)
                    # In production, you'd save the mask separately
                    try:
                        img = Image.open(image_path)
                        _, mask = inference.infer_image(img)
                        
                        if mask is not None:
                            st.markdown("**Probability Map:**")
                            st.image(mask, caption="Manipulation Heatmap", use_container_width=True)
                    except:
                        pass
