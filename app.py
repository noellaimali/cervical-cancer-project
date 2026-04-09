import streamlit as st
# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Cervical Cancer Prediction Model",
    page_icon="🔬",
    layout="wide"
)

import tensorflow as tf
import sqlite3
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import auth
import time
import db_init

# Initialize database
db_init.init_db()

# Ensure a default admin existed for deployment testing
try:
    auth.register_user("admin", "admin123", email="admin@cytoscan.com", phone="000")
    # Promote to admin if not already
    conn = sqlite3.connect('cytoscan.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = 'admin' WHERE username = 'admin'")
    conn.commit()
    conn.close()
except:
    pass

# 1. Page Config (Moved to Top)

# 2. CONSTANTS
MODEL_PATH = 'cervical_cell_classifier.h5'
CLASSES_PATH = 'classes.json'
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 3. SESSION STATE
if 'user' not in st.session_state:
    st.session_state.user = None
if 'img_index' not in st.session_state:
    st.session_state.img_index = 0

# 4. UTILITY FUNCTIONS
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_as_page_bg(bin_file):
    if not os.path.exists(bin_file):
        return
    bin_str = get_base64_of_bin_file(bin_file)
    ext = bin_file.split('.')[-1].lower()
    mime = f"image/{ext}" if ext != 'jpg' else 'image/jpeg'
    st.markdown(f'''
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("data:{mime};base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    ''', unsafe_allow_html=True)

@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = []
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r') as f:
            classes = json.load(f)
    else:
        classes = ["NORMAL", "ABNORMAL"]
    return model, classes

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def is_valid_medical_image(image):
    img_hsv = image.convert('HSV')
    np_hsv = np.array(img_hsv)
    h, s = np_hsv[:,:,0], np_hsv[:,:,1]
    high_sat_mask = s > 40 
    if np.sum(high_sat_mask) / h.size < 0.15:
        return True, "Grayscale/Low saturation"
    
    valid_hues = ((h > 100) & (h < 250) | (h < 15) | (h > 240))
    invalid_hues = (h > 25) & (h < 100)
    v_pixels = np.sum(high_sat_mask & valid_hues)
    i_pixels = np.sum(high_sat_mask & invalid_hues)
    
    if i_pixels > v_pixels * 1.5:
        return False, "Non-medical color palette detected."
    return True, "Valid morphology colors."

# 5. MAIN APPLICATION
def main():
    # --- PHASE A: AUTHENTICATION GATE ---
    if st.session_state.user is None:
        # Dynamic Slideshow Logic
        images = []
        if os.path.exists("images"):
            images = [os.path.join("images", f) for f in os.listdir("images") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not images:
            images = ["logo.jpg"] if os.path.exists("logo.jpg") else []

        # Generate CSS animation dynamically
        if images:
            keyframes = []
            n_images = len(images)
            for idx, img_path in enumerate(images):
                b64 = get_base64_of_bin_file(img_path)
                pct = int((idx / n_images) * 100)
                ext = img_path.split('.')[-1].lower()
                mime = f"image/{ext}" if ext != 'jpg' else 'image/jpeg'
                keyframes.append(f"{pct}% {{ background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('data:{mime};base64,{b64}'); }}")
            
            # Loop back to 100%
            first_b64 = get_base64_of_bin_file(images[0])
            ext = images[0].split('.')[-1].lower()
            mime = f"image/{ext}" if ext != 'jpg' else 'image/jpeg'
            keyframes.append(f"100% {{ background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('data:{mime};base64,{first_b64}'); }}")
            
            keyframes_str = "\n".join(keyframes)
            anim_duration = n_images * 4  # 4 seconds per image
            
            st.markdown(f'''
            <style>
            @keyframes bgSlideshow {{
                {keyframes_str}
            }}
            .stApp {{
                animation: bgSlideshow {anim_duration}s infinite linear;
                background-size: cover; 
                background-position: center; 
            }}
            [data-testid="stSidebar"], [data-testid="stSidebarNav"], header, #MainMenu {{ display: none !important; }}
            .login-card {{
                background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(15px); padding: 40px;
                border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.4);
                max-width: 450px; margin: auto; margin-top: 10vh; border: 1px solid rgba(255, 255, 255, 0.3);
            }}
            </style>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <style>
            [data-testid="stSidebar"], [data-testid="stSidebarNav"], header, #MainMenu { display: none !important; }
            .login-card {
                background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(15px); padding: 40px;
                border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.4);
                max-width: 450px; margin: auto; margin-top: 10vh; border: 1px solid rgba(255, 255, 255, 0.3);
            }
            </style>
            ''', unsafe_allow_html=True)

        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #1a252f; margin-bottom: 0;'>Cervical Cancer System</h2><p style='text-align: center; color: #576574; margin-bottom: 20px;'>Portal Authentication</p>", unsafe_allow_html=True)
        
        t1, t2 = st.tabs(["🔒 Secure Login", "📝 New Account"])
        with t1:
            u = st.text_input("Username", key="l_user")
            p = st.text_input("Password", type="password", key="l_pass")
            if st.button("Unlock Dashboard", use_container_width=True):
                user = auth.login_user(u, p)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        with t2:
            nu = st.text_input("New Username", key="r_user")
            ne = st.text_input("Email", key="r_email")
            nph = st.text_input("Phone Number", key="r_phone")
            npw = st.text_input("Password", type="password", key="r_pass")
            if st.button("Initialize Account ✨", use_container_width=True):
                if auth.register_user(nu, npw, ne, nph):
                    st.success("Success! Please log in.")
                else:
                    st.error("Username taken.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.stop()

    # --- PHASE B: DASHBOARD ---
    if os.path.exists("background.jpg"):
        set_bg_as_page_bg("background.jpg")

    # Global CSS for Dashboard
    st.markdown("""
        <style>
        h1, h2, h3, p, span, label { color: #e0e6ed !important; }
        .stButton>button { background-color: #1e88e5; color: white; border-radius: 8px; }
        .prediction-card { padding: 24px; border-radius: 16px; background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; }
        .history-card { padding: 12px; border-left: 4px solid #1e88e5; background: rgba(13, 17, 23, 0.6); margin-bottom: 10px; border-radius: 0 8px 8px 0; font-size: 0.85em; }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("🔐 Secure Session")
    st.sidebar.write(f"Active: **{st.session_state.user[1]}**")
    if st.sidebar.button("Logout System", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")
    
    nav_options = ["Home", "Prediction", "Patient Reports", "Model Info", "User Profile", "About"]
    if len(st.session_state.user) > 3 and st.session_state.user[3] == 'admin':
        nav_options.append("Admin Dashboard")
        
    page = st.sidebar.selectbox("Navigate Workspace", nav_options)
    
    st.sidebar.markdown("---")
    model, classes = load_model_and_classes()
    if model:
        st.sidebar.success(f"✅ System Online\n\nClasses: {', '.join(classes)}")
    else:
        st.sidebar.error("❌ Model Offline")

    user_history = auth.get_user_history(st.session_state.user[0])
    if user_history:
        st.sidebar.subheader("Recent History")
        for h in user_history[:5]:
            p_name = h[4] if (len(h) > 4 and h[4]) else "Unknown Patient"
            st.sidebar.markdown(f'<div class="history-card"><strong>{h[1]} ({p_name})</strong><br><small>{h[2]:.1%} | {h[3][:16]}</small></div>', unsafe_allow_html=True)

    # Header
    c1, c2 = st.columns([1, 4])
    with c1:
        if os.path.exists("logo.jpg"):
            st.image("logo.jpg", width=120)
        else:
            st.title("🔬")
    with c2:
        st.title("CytoScan: AI Diagnostic Platform")
    st.markdown("---")

    # Routing
    if page == "Home":
        st.header("Home: Clinical Diagnostic Suite")
        st.markdown("""
        ### Revolutionary Cervical Screening
        Welcome to **CytoScan**, an advanced pathological screening tool powered by deep learning. This platform is designed to assist medical researchers and cytotechnologists in identifying cellular abnormalities with high precision.

        #### ⚡ Core Capabilities:
        - **Real-time Diagnostics**: Immediate feedback on single microscopic cell samples.
        - **Batch Screening**: Process high volumes of patient data in seconds.
        - **Persistence**: All predictions are securely logged to your clinical history.
        - **Validation**: Intelligent color-masking ensures only valid medical images are analyzed.

        *Select a tool from the sidebar to begin.*
        """)
        st.info("📊 **System Tip:** For optimal accuracy, ensure cell images are captured at 40x magnification and centered.")

    elif page == "Prediction":
        st.subheader("📋 Patient Details")
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient Name", key="pname")
        with c2:
            patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30, key="page")
            
        st.markdown("---")
        t1, t2 = st.tabs(["Single Image Diagnostic", "Batch Processing"])
        with t1:
            up = st.file_uploader("Upload cell image...", type=["png", "jpg", "jpeg", "bmp"], key="s")
            if up:
                img = Image.open(up)
                c1, _, c2 = st.columns([1, 0.1, 1.2])
                with c1: st.image(img, use_container_width=True, caption="Microscope Sample")
                with c2:
                    if st.button("Run Diagnostic ✨"):
                        valid, reason = is_valid_medical_image(img)
                        if not valid:
                            st.warning(f"Rejected: {reason}")
                        else:
                            proc = preprocess_image(img)
                            pred = model.predict(proc, verbose=0)
                            
                            # Handle Multiclass Logic (using Argmax)
                            idx = np.argmax(pred)
                            conf = pred[0][idx]
                            res = classes[idx]
                            
                            # Normal/Abnormal/Unknown Mapping Based on Class Name and Confidence
                            if conf < 0.65 or "invalid" in res.lower():
                                final, color = "Unknown", "#f39c12"
                            elif "non-cancerous" in res.lower() or "normal" in res.lower():
                                final, color = "Normal", "#27ae60"
                            else:
                                final, color = "Abnormal", "#e74c3c"
                            
                            auth.save_prediction(st.session_state.user[0], up.name, final, float(conf), patient_name, patient_age)
                            
                            st.markdown(f'''
                            <div class="prediction-card" style="border-top: 5px solid {color}; padding-bottom: 5px;">
                                <h1 style="color: {color}; margin-top: 0; font-size: 2.2em;">{final}</h1>
                                <p style="font-size: 1.1em; color: #8b949e; margin-bottom: 0;">Diagnostic Confidence</p>
                                <h3 style="margin-top: 0;">{conf:.1%}</h3>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Add Probability Distribution Chart
                            st.markdown("#### Category Distribution")
                            prob_df = pd.DataFrame({'Category': classes, 'Probability (%)': [float(p*100) for p in pred[0]]})
                            fig_probs = px.bar(prob_df, x='Category', y='Probability (%)', color='Category', 
                                              range_y=[0, 100], text_auto='.1f',
                                              color_discrete_map={'CANCEROUS': '#e74c3c', 'NON-CANCEROUS': '#27ae60', 'INVALID': '#6c757d'})
                            fig_probs.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': 'white'}, height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
                            st.plotly_chart(fig_probs, use_container_width=True)

        with t2:
            bups = st.file_uploader("Upload multiple samples...", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True, key="b")
            if bups and st.button("Start Batch Analysis"):
                results = []
                for bup in bups:
                    img = Image.open(bup)
                    val, _ = is_valid_medical_image(img)
                    if not val:
                        results.append({"File": bup.name, "Result": "Unknown", "Conf": "N/A"})
                        continue
                    proc = preprocess_image(img)
                    pred = model.predict(proc, verbose=0)
                    
                    # Handle Multiclass Logic (using Argmax)
                    idx = np.argmax(pred)
                    conf = pred[0][idx]
                    res = classes[idx]
                    
                    # Normal/Abnormal/Unknown Mapping
                    if conf < 0.65 or "invalid" in res.lower():
                        final_status = "Unknown"
                    elif "non-cancerous" in res.lower() or "normal" in res.lower():
                        final_status = "Normal"
                    else:
                        final_status = "Abnormal"
                        
                    auth.save_prediction(st.session_state.user[0], bup.name, final_status, float(conf), patient_name, patient_age)
                    results.append({"File": bup.name, "Result": final_status, "Conf": f"{conf:.1%}"})
                st.table(pd.DataFrame(results))

    elif page == "Model Info":
        st.header("🔬 CytoScan Intelligence Details")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            #### Neural Architecture
            The system utilizes a custom **Convolutional Neural Network (CNN)** optimized for morphological feature extraction.
            - **Input Scale**: 128x128 Pixels (RGB)
            - **Layers**: Multiple Conv2D blocks with ReLU activation and Dropout for regularization.
            - **Classes**: Categorical Cross-Entropy (Multiclass).
            """)
        with col_m2:
            st.markdown("""
            #### Operational Metrics
            - **Optimizer**: Adam (Adaptive Moment Estimation)
            - **Performance**: Sub-100ms inference on standard CPUs.
            - **Accuracy Basis**: Trained on standardized clinical datasets (Herlev & SIPaKMeD).
            """)
        st.markdown("---")
        st.subheader("Technical Source Overview")
        st.code("""
        # Model Structure (Summary)
        Model: CytoScan_CNN_v1
        Layers: Conv2D(32), MaxPooling, Conv2D(64), Flatten, Dense(512), Dense(3)
        Activation: Softmax (Output Layer)
        """)

    elif page == "User Profile":
        st.header("👤 User Account")
        u = st.session_state.user
        ph = u[4] if len(u)>4 else "Not Provided"
        st.write(f"**Username:** {u[1]}")
        st.write(f"**Email:** {u[2]}")
        st.write(f"**Phone:** {ph}")
        st.markdown("---")
        h = auth.get_user_history(u[0])
        if h: st.dataframe(pd.DataFrame(h, columns=["Image", "Result", "Confidence", "Time", "Patient Name", "Age"]), use_container_width=True)

    elif page == "About":
        st.header("About CytoScan")
        st.write("CytoScan is an experimental AI platform for early detection of abnormalities in cervical cell slides.")
        st.warning("**Disclaimer:** For educational use only. Not for clinical diagnosis.")

    elif page == "Patient Reports":
        st.header("🗂️ Patient Reports")
        st.write("Generate and download individual patient diagnostic reports.")
        
        is_admin = len(st.session_state.user) > 3 and st.session_state.user[3] == 'admin'
        u_id = None if is_admin else st.session_state.user[0]
        
        patients = auth.get_unique_patients(u_id)
        
        if not patients:
            st.info("No patient records found in your history.")
        else:
            selected_patient = st.selectbox("Select Patient", ["-- All Patients --"] + patients)
            
            report_data = auth.get_patient_report(u_id)
            if report_data:
                df_report = pd.DataFrame(report_data, columns=["Patient Name", "Age", "Image Processed", "Result", "Confidence", "Timestamp"])
                
                # Format confidence
                df_report['Confidence'] = df_report['Confidence'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else x)
                
                # Filter by selection
                if selected_patient != "-- All Patients --":
                    df_report = df_report[df_report["Patient Name"] == selected_patient]
                
                if not df_report.empty:
                    st.subheader(f"Diagnostic History: {selected_patient}")
                    st.dataframe(df_report, use_container_width=True, hide_index=True)
                    
                    # Download CSV Header
                    csv = df_report.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv,
                        file_name=f"report_{selected_patient.replace(' ', '_')}.csv" if selected_patient != "-- All Patients --" else "all_patients_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info(f"No records found for {selected_patient}.")

    elif page == "Admin Dashboard":
        st.header("🛠️ Admin Dashboard")
        st.write("Welcome to the Administration Panel. Here you can monitor system activity.")
        
        t1, t2 = st.tabs(["👥 Users Directory", "📊 System Predictions"])
        with t1:
            st.subheader("Registered Users")
            users = auth.get_all_users()
            if users:
                df_users = pd.DataFrame(users, columns=["ID", "Username", "Email", "Role", "Phone", "Created At"])
                st.dataframe(df_users, use_container_width=True, hide_index=True)
            else:
                st.write("No users found.")
                
        with t2:
            st.subheader("All Predictions History")
            preds = auth.get_all_predictions()
            if preds:
                # Format the confidence as percentage
                formatted_preds = [(p[0], p[1], p[2], p[3], f"{p[4]:.1%}", p[5], p[6], p[7]) for p in preds]
                df_preds = pd.DataFrame(formatted_preds, columns=["ID", "Username", "Image", "Result", "Confidence", "Timestamp", "Patient Name", "Age"])
                st.dataframe(df_preds, use_container_width=True, hide_index=True)
            else:
                st.write("No predictions found.")

if __name__ == "__main__":
    main()
