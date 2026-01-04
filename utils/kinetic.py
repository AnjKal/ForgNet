import streamlit as st
import streamlit.components.v1 as components
import os

def load_css():
    """Loads the global CSS file and injects it into the app."""
    css_path = os.path.join("assets", "style.css")
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def inject_kinetic_tracking():
    """
    Injects JS to track mouse position and update CSS variables.
    This creates the 'reactive' background and glowing borders.
    """
    js_code = """
    <script>
    document.addEventListener('mousemove', (e) => {
        const doc = document.documentElement;
        // Calculate percentage for gradients
        const x = (e.clientX / window.innerWidth) * 100;
        const y = (e.clientY / window.innerHeight) * 100;
        
        doc.style.setProperty('--mouse-x', x + '%');
        doc.style.setProperty('--mouse-y', y + '%');
    });

    // Optional: Add tilt effect to specific elements if needed
    // This is a global listener for generic 'kinetic-tilt' class
    document.addEventListener('mousemove', (e) => {
        document.querySelectorAll('.kinetic-tilt').forEach(card => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Basic tracking for internal glows
            card.style.setProperty('--card-mouse-x', `${x}px`);
            card.style.setProperty('--card-mouse-y', `${y}px`);
        });
    });
    </script>
    """
    components.html(js_code, height=0, width=0)

def kinetic_header(title, subtitle):
    st.markdown(f"""
    <div class="glass-panel" style="text-align: center; margin-bottom: 2rem;">
        <div class="subtitle-text">{subtitle}</div>
        <div class="title-text">{title}</div>
    </div>
    """, unsafe_allow_html=True)
