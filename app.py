import streamlit as st
import time
from utils import kinetic

from components import analysis
from components import network
from components import navigation_graph
from components import explainability
from components import history
from utils import inference


# 1. Page Config (Must be first)
st.set_page_config(
    page_title="Forensic AI Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Kinetic Core
kinetic.load_css()
kinetic.inject_kinetic_tracking()

# 3. Sidebar Navigation
st.sidebar.markdown("## 👁️ FORENSIC STUDIO")
nav_options = ["Upload & Scan", "Statistics", "Explainability", "Graph Intelligence", "History", "Settings"]

# Check Query Params for Navigation (Graph Interaction)
qp = st.query_params
default_index = 0
if "page" in qp:
    target_page = qp["page"]
    if target_page in nav_options:
        default_index = nav_options.index(target_page)

# Render Graph Navigation
navigation_graph.render_sidebar_graph(nav_options[default_index])

st.sidebar.markdown("### Manual Override")
selected_page = st.sidebar.radio("Navigation", nav_options, index=default_index, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:** 🟢 *Online*")
st.sidebar.markdown("**GPU:** *NVIDIA A100 (Mock)*")

# 4. Main Router
if selected_page == "Upload & Scan":
    kinetic.kinetic_header("EVIDENCE INGESTION", "SECURE DROPZONE")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-panel" style="min-height: 400px; display: flex; align-items: center; justify-content: center; flex-direction: column; border-style: dashed;">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop Suspect Media Here", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            st.success("File Ingested. Hash Verification Complete.")
            
            # Save uploaded file
            from pathlib import Path
            from datetime import datetime
            import os
            
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = Path(uploaded_file.name).suffix
            case_id = f"{timestamp}_{uploaded_file.name.replace(file_ext, '')}"
            save_path = uploads_dir / f"{case_id}{file_ext}"
            
            # Save file
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run inference
            with st.spinner("Running Forgery Detection Model..."):
                from PIL import Image
                import numpy as np
                
                image = Image.open(save_path).convert("RGB")
                label, mask = inference.infer_image(image)
            
            # Display results INSIDE the box
            st.markdown('<div style="width: 100%; margin_top: 1rem;">', unsafe_allow_html=True)
            col_img, col_mask = st.columns(2)
            
            with col_img:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            with col_mask:
                if mask is not None:
                    st.markdown("**Probability Map**")
                    st.image(mask, use_container_width=True)
                else:
                    st.markdown("**No Manipulation Detected**")
                    st.info("Image appears authentic")
            
            # Show verdict
            if label == "authentic":
                st.success(f"✓ VERDICT: AUTHENTIC")
                annotation = "authentic"
            elif label == "forged":
                st.error(f"⚠ VERDICT: FAKE/MANIPULATED")
                # Encode mask to RLE
                if mask is not None:
                    mask_array = np.array(mask)
                    binary_mask = (mask_array > 128).astype(np.uint8)
                    annotation = inference.rle_encode(binary_mask)
                else:
                    annotation = ""
            else:
                st.warning("Error during inference")
                annotation = ""
            st.markdown('</div>', unsafe_allow_html=True)


            # Generate embedding for graph linkage (using DINOv2)
            embedding = inference.get_embedding(image)

            # Save to history with embedding
            annotation = inference.save_prediction(case_id, str(save_path), label, mask, embedding=embedding)
            
            st.snow()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 📋 Quick Stats")
        st.markdown(f"**Session ID:** {int(time.time())}")
        st.markdown("**Security Level:** 🔒 Top Secret")
        st.markdown("**Model Version:** v4.2.0-Alpha")
        st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "Statistics":
    kinetic.kinetic_header("STATISTICS", "SYSTEM INSIGHTS")
    
    import glob
    
    # Calculate stats
    hist = inference.get_history()
    total_scans = len(hist)
    authentic_count = sum(1 for h in hist if h["label"] == "authentic")
    forged_count = sum(1 for h in hist if h["label"] == "forged")
    
    # Count explainability images (*_gradcam.png in current dir)
    xai_images = len(glob.glob("*_gradcam.png"))
    
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scans", total_scans)
    with col2:
        st.metric("Authentic", authentic_count)
    with col3:
        st.metric("Forged", forged_count)
    with col4:
        st.metric("XAI Generated", xai_images)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Distribution")
    if total_scans > 0:
        st.bar_chart({"Authentic": authentic_count, "Forged": forged_count})
    else:
        st.info("No data available yet.")

elif selected_page == "Explainability":
    kinetic.kinetic_header("XAI EXPLORER", "MODEL REASONING")
    explainability.render_explainability()

elif selected_page == "Graph Intelligence":
    kinetic.kinetic_header("CASE NETWORK", "VISUAL SIMILARITY AI")
    
    # 1. Load History
    history_data = inference.get_history()
    
    if not history_data:
        st.info("No cases analyzed yet. Upload images to build the network.")
    else:
        # 2. Build Graph
        # We need cosine similarity between all pairs.
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        nodes = []
        edges = []
        
        # Filter records with embeddings
        records = [h for h in history_data if "embedding" in h]
        
        if len(records) < 2:
            st.warning("Not enough data points with embeddings to build connections.")
            # Show single node if 1 exists
            for rec in records:
                nodes.append({
                    "id": rec["case_id"],
                    "label": f"{rec['label'].upper()}\n{rec['case_id'][:8]}...",
                    "color": "#ff4b4b" if rec["label"] == "forged" else "#00cc96",
                    "shape": "dot"
                })
        else:
            # Extract matrix
            embeddings = np.array([r["embedding"] for r in records])
            ids = [r["case_id"] for r in records]
            labels = [r["label"] for r in records]
            
            # Create Nodes
            for i, case_id in enumerate(ids):
                nodes.append({
                    "id": case_id,
                    "label": f"{labels[i].upper()}\n{case_id[:8]}...",
                    "title": f"Case: {case_id}\nVerdict: {labels[i]}", # Tooltip
                    "color": "#ff4b4b" if labels[i] == "forged" else "#00cc96",
                    "shape": "dot",
                    "size": 20
                })
            
            # Compute Similarity
            sim_matrix = cosine_similarity(embeddings)
            
            # Create Edges (threshold > 0.85)
            # Avoid duplicate edges (i, j) and (j, i)
            threshold = 0.85
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    score = sim_matrix[i][j]
                    if score > threshold:
                        edges.append({
                            "from": ids[i],
                            "to": ids[j],
                            "width": (score - threshold) * 8, # Thicker for higher sim
                            "title": f"Similarity: {score:.2f}",
                            "color": {"color": "#bd00ff", "opacity": 0.6}
                        })
        
        # 3. Render
        st.markdown(f"**Network Stats:** {len(nodes)} Nodes | {len(edges)} Visual Links")
        network.render_network_graph(nodes, edges)

elif selected_page == "History":
    kinetic.kinetic_header("CASE HISTORY", "PAST INVESTIGATIONS")
    history.render_history()

elif selected_page == "Settings":
    kinetic.kinetic_header("SYSTEM CONFIG", "ADVANCED PARAMETERS")
