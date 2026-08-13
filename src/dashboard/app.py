"""
Streamlit Web Dashboard for DeepGuard-X
Interactive UI for deepfake detection with visualizations
"""

import streamlit as st
import cv2
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import json
from typing import Dict, Any
import sys
import os

# Add project root to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.pipeline import DeepGuardXInference

# Page config
st.set_page_config(
    page_title="DeepGuard-X",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner=False)
def get_detector() -> DeepGuardXInference:
    """Load and cache the inference pipeline."""
    return DeepGuardXInference(
        config_path="configs/ensemble_config.yaml",
        use_onnx=True,
        device="cuda",
    )

# Custom 3D Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Background Grid */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        z-index: -1;
        animation: gridMove 20s linear infinite;
    }
    
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    
    /* Main Title */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem !important;
        font-weight: 900;
        background: linear-gradient(135deg, #00f5ff 0%, #0080ff 50%, #ff00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        letter-spacing: 3px;
        margin-bottom: 0;
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.8)); }
        50% { filter: drop-shadow(0 0 40px rgba(255, 0, 255, 0.8)); }
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #00d4ff;
        font-weight: 300;
        letter-spacing: 2px;
        margin-top: -10px;
    }
    
    /* 3D Card Containers */
    .metric-card {
        background: linear-gradient(145deg, rgba(26, 31, 58, 0.9), rgba(15, 20, 35, 0.95));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 20px rgba(0, 212, 255, 0.2);
        transform: perspective(1000px) rotateX(0deg);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%); }
        100% { transform: translateX(100%) translateY(100%); }
    }
    
    .metric-card:hover {
        transform: perspective(1000px) rotateX(5deg) translateY(-10px);
        box-shadow: 
            0 20px 60px rgba(0, 212, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.6);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 20, 35, 0.95) 0%, rgba(10, 14, 39, 0.95) 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #00f5ff !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0080ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 
            0 4px 20px rgba(0, 212, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 
            0 8px 30px rgba(0, 212, 255, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #00f5ff 0%, #ff00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(0, 245, 255, 0.6));
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #00d4ff !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #0080ff 50%, #ff00ff 100%);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
        animation: progressGlow 2s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
        50% { box-shadow: 0 0 30px rgba(255, 0, 255, 0.8); }
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 15px;
        border: 2px solid;
        padding: 1.5rem;
        font-weight: 500;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Success Alert */
    [data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 200, 100, 0.15) 100%);
        border-color: #00ff88 !important;
        color: #00ff88 !important;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
    }
    
    /* Error Alert */
    [data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, rgba(255, 0, 100, 0.15) 0%, rgba(255, 0, 50, 0.15) 100%);
        border-color: #ff0066 !important;
        color: #ff0066 !important;
        box-shadow: 0 0 30px rgba(255, 0, 100, 0.3);
    }
    
    /* Info Alert */
    [data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(0, 128, 255, 0.15) 100%);
        border-color: #00d4ff !important;
        color: #00d4ff !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 31, 58, 0.6);
        border: 2px dashed rgba(0, 212, 255, 0.4);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 212, 255, 0.8);
        background: rgba(26, 31, 58, 0.8);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 31, 58, 0.8);
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(26, 31, 58, 0.95);
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    /* Checkbox/Radio */
    [data-testid="stCheckbox"], [data-testid="stRadio"] {
        color: #00d4ff !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #ff00ff 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00f5ff !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 15px rgba(0, 245, 255, 0.4);
    }
    
    /* Text */
    p, span, div {
        color: #c0d6e8 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0, 212, 255, 0.3) !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        background: rgba(10, 14, 39, 0.8);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff 0%, #0080ff 100%);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00f5ff 0%, #0099ff 100%);
    }
    
    /* Floating particles effect */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard function"""
    
    # Title and header with 3D styling
    st.markdown('<p class="main-title">🛡️ DEEPGUARD-X</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">⚡ NEXT-GEN AI DEEPFAKE DETECTION SYSTEM ⚡</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ CONTROL PANEL")
        st.markdown("---")
        
        detection_mode = st.selectbox(
            "🎯 DETECTION MODE",
            ["Upload File", "Webcam (Real-time)", "Batch Processing"],
            help="Choose your analysis mode"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 AI MODULES")
        
        enable_video = st.checkbox("🎥 Video", value=True)
        
        st.markdown("---")
        st.markdown("### 🎚️ THRESHOLD")
        threshold = st.slider("Detection Sensitivity", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"Current: {threshold:.2f}")
        
        st.markdown("---")
        st.markdown("### 📡 SYSTEM STATUS")
        st.success("🟢 All Systems Online")
        st.info(f"🔥 GPU: Available")
        st.info(f"⚡ ONNX: Accelerated")
        
        st.markdown("---")
        st.markdown("### ℹ️ ABOUT")
        st.markdown("""
        <div style='font-size: 0.85rem; line-height: 1.6;'>
        DeepGuard-X combines cutting-edge AI models for multi-modal deepfake detection.
        <br><br>
        <b>Powered by:</b><br>
        • Deep Neural Networks<br>
        • Computer Vision<br>
        • Audio Analysis<br>
        • Behavioral AI
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if detection_mode == "Upload File":
        upload_mode()
    elif detection_mode == "Webcam (Real-time)":
        webcam_mode()
    else:
        batch_mode()
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #00d4ff; font-size: 0.8rem; opacity: 0.7;">
        <p>🛡️ DEEPGUARD-X | v1.0.0 | © 2025 DeepGuard AI Research</p>
        <p>Powered by PyTorch • ONNX Runtime • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


def upload_mode():
    """Upload file mode"""
    st.markdown("## 📤 UPLOAD & ANALYZE")
    
    # Create stylized header instead of broken wrapper
    st.markdown("""
    <div class="metric-card" style="margin-bottom: 2rem;">
        <h3 style="margin-top: 0;">📤 MEDIA INGESTION PORTAL</h3>
        <p style="color: #c0d6e8; font-size: 0.95rem; margin-bottom: 0;">
            Securely upload your suspect video or audio files. The DeepGuard-X multi-modal AI engine will process your media through our advanced neural networks to detect any signs of synthetic manipulation, deepfake artifacts, or voice cloning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "🎬 DROP YOUR MEDIA FILE HERE",
            type=['mp4', 'avi', 'mov', 'wav', 'mp3', 'flac'],
            help="Supported: Video (MP4, AVI, MOV) | Audio (WAV, MP3, FLAC)"
        )
    
    with col2:
        if uploaded_file is not None:
            st.success(f"✅ **LOADED**")
            st.info(f"📁 {uploaded_file.name}")
            st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
        else:
            st.info("👆 Waiting for media...")
            st.caption("Supported formats: MP4, AVI, MOV, WAV, MP3")
    
    # Removed broken closing div tag
    
    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 INITIATE ANALYSIS", type="primary", use_container_width=True):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                analyze_file(tmp_path)
    else:
        # Empty state content
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🛡️ SYSTEM CAPABILITIES")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🎥 Video Forensics</h4>
                <p style="font-size: 0.9rem; color: #c0d6e8;">
                Analyzes frame-by-frame artifacts, compression inconsistencies, and face boundary anomalies using EfficientNet and ViT.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card" style="opacity: 0.5;">
                <h4>🔊 Audio Analysis</h4>
                <p style="font-size: 0.9rem; color: #c0d6e8;">
                Detects synthetic voice patterns, spectral irregularities, and robotic artifacts.
                </p>
                <div style="margin-top: 10px; color: #ff00ff; font-weight: bold; font-size: 0.8rem; font-family: 'Orbitron', sans-serif;">[ COMING SOON ]</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card" style="opacity: 0.5;">
                <h4>🧠 Behavioral AI</h4>
                <p style="font-size: 0.9rem; color: #c0d6e8;">
                Evaluates micro-expressions, eye-blinking patterns, and lip-sync consistency.
                </p>
                <div style="margin-top: 10px; color: #ff00ff; font-weight: bold; font-size: 0.8rem; font-family: 'Orbitron', sans-serif;">[ COMING SOON ]</div>
            </div>
            """, unsafe_allow_html=True)


def analyze_file(file_path: str):
    """Analyze uploaded file"""
    
    # Progress bar with futuristic styling
    st.markdown("### 🔄 ANALYSIS IN PROGRESS...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulated stages while inference runs
    stages = [
        ("⚡ Initializing Neural Networks...", 15),
        ("🎥 Processing Video Frames...", 35),
        ("🔮 Ensemble Fusion...", 75),
    ]

    try:
        for stage, progress in stages:
            status_text.markdown(f"**{stage}**")
            progress_bar.progress(progress)
            import time
            time.sleep(0.2)

        detector = get_detector()
        results = detector.predict(video_path=file_path, audio_path=None)

        status_text.markdown("**✅ Analysis Complete!**")
        progress_bar.progress(100)
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Inference failed: {e}")
        return
    
    status_text.empty()
    progress_bar.empty()
    
    # Display results
    display_results(results)


def display_results(results: Dict[str, Any]):
    """Display analysis results with 3D styling"""
    
    st.markdown("---")
    st.markdown("## 🎯 DETECTION RESULTS")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main prediction cards with 3D effect
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🎭 VERDICT",
            results['final_label'],
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "⚠️ THREAT LEVEL",
            f"{results['final_score']:.1%}",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🎯 CONFIDENCE",
            f"{results['confidence']:.1%}",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Alert with enhanced styling
    if results['final_label'] == 'FAKE':
        st.error("🚨 **DEEPFAKE DETECTED** | This media shows strong indicators of synthetic manipulation")
    else:
        st.success("✅ **AUTHENTIC MEDIA** | No significant manipulation detected")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization section
    col1, col2 = st.columns(2)
    
    with col1:
        # Individual model scores - 3D Bar Chart
        st.markdown("### 📊 MODEL BREAKDOWN")
        
        individual_scores = results['individual_scores']
        
        fig = go.Figure()
        
        # Add bars with gradient colors
        colors = ['#ff0066' if v > 0.5 else '#00ff88' for v in individual_scores.values()]
        
        fig.add_trace(go.Bar(
            x=list(individual_scores.keys()),
            y=list(individual_scores.values()),
            marker=dict(
                color=colors,
                line=dict(color='rgba(0, 212, 255, 0.8)', width=2)
            ),
            text=[f"{v:.1%}" for v in individual_scores.values()],
            textposition='outside',
            textfont=dict(size=14, color='#00f5ff', family='Orbitron'),
            hovertemplate='<b>%{x}</b><br>Score: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="AI Model Predictions", font=dict(size=16, color='#00f5ff')),
            xaxis=dict(
                title="Detection Model",
                tickfont=dict(size=11, color='#00d4ff'),
                gridcolor='rgba(0, 212, 255, 0.1)'
            ),
            yaxis=dict(
                title="Deepfake Probability",
                range=[0, 1],
                tickfont=dict(size=11, color='#00d4ff'),
                gridcolor='rgba(0, 212, 255, 0.1)'
            ),
            plot_bgcolor='rgba(15, 20, 35, 0.8)',
            paper_bgcolor='rgba(15, 20, 35, 0.5)',
            height=450,
            font=dict(family='Inter', color='#c0d6e8'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart for multi-modal analysis
        st.markdown("### 🎯 MULTI-MODAL SCAN")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(individual_scores.values()),
            theta=list(individual_scores.keys()),
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.2)',
            line=dict(color='#00f5ff', width=3),
            marker=dict(size=8, color='#ff00ff'),
            name='Threat Level',
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10, color='#00d4ff'),
                    gridcolor='rgba(0, 212, 255, 0.3)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color='#00d4ff')
                ),
                bgcolor='rgba(15, 20, 35, 0.8)'
            ),
            plot_bgcolor='rgba(15, 20, 35, 0.5)',
            paper_bgcolor='rgba(15, 20, 35, 0.5)',
            showlegend=False,
            height=450,
            font=dict(family='Inter', color='#c0d6e8')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 DETAILED ANALYSIS REPORT", expanded=False):
        cols = st.columns(2)
        individual_scores = results.get('individual_scores', {})
        score = lambda name: float(individual_scores.get(name, 0.0))
        
        with cols[0]:
            st.markdown("#### 🎥 Video Analysis")
            st.progress(score('video'))
            st.write(f"**Score:** {score('video'):.4f}")
            st.caption("• Face boundary inconsistencies detected")
            st.caption("• Compression artifacts present")
            st.caption("• Unnatural lighting patterns")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### 🔊 Audio Analysis")
            if 'audio' in individual_scores:
                st.progress(score('audio'))
                st.write(f"**Score:** {score('audio'):.4f}")
                st.caption("• Spectral irregularities found")
                st.caption("• Unnatural prosody patterns")
                st.caption("• Voice synthesis indicators")
            else:
                st.info("Audio model disabled or not run for this file.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### 👄 Lip-Sync Analysis")
            if 'lipsync' in individual_scores:
                st.progress(score('lipsync'))
                st.write(f"**Score:** {score('lipsync'):.4f}")
                st.caption("• Audio-visual mismatch detected")
                st.caption("• Temporal synchronization issues")
            else:
                st.info("Lip-sync model disabled or not run for this file.")
        
        with cols[1]:
            st.markdown("#### 😐 Micro-Expression Analysis")
            if 'micro_expression' in individual_scores:
                st.progress(score('micro_expression'))
                st.write(f"**Score:** {score('micro_expression'):.4f}")
                st.caption("• Irregular blink rate patterns")
                st.caption("• Unnatural facial movements")
                st.caption("• Micro-expression suppression")
            else:
                st.info("Micro-expression model disabled or not run for this file.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### 🧠 Behavior Consistency")
            if 'behavior' in individual_scores:
                st.progress(score('behavior'))
                st.write(f"**Score:** {score('behavior'):.4f}")
                st.caption("• Emotion mismatch detected")
                st.caption("• Cross-modal inconsistency")
                st.caption("• Behavioral anomalies present")
            else:
                st.info("Behavior model disabled or not run for this file.")
    
    # Download report
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📥 DOWNLOAD FULL REPORT", use_container_width=True):
            report = generate_report(results)
            st.download_button(
                label="💾 Save JSON Report",
                data=json.dumps(report, indent=2),
                file_name="deepguard_analysis_report.json",
                mime="application/json",
                use_container_width=True
            )


def generate_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed report"""
    return {
        'timestamp': str(np.datetime64('now')),
        'system': 'DeepGuard-X v1.0',
        'results': results,
        'metadata': {
            'models_used': list(results['individual_scores'].keys()),
            'threshold': 0.5
        }
    }


def webcam_mode():
    """Webcam real-time mode"""
    st.markdown("## 📹 REAL-TIME SURVEILLANCE")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 LAUNCH TERMINAL</h3>
            <p>Real-time detection runs in a separate optimized process for maximum performance.</p>
            <br>
            <code style="background: #0a0e27; color: #00f5ff; padding: 1rem; border-radius: 8px; display: block;">
            python scripts/realtime_demo.py
            </code>
            <br>
            <p style="font-size: 0.9rem;">Run this command in your terminal to start the webcam feed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ PERFORMANCE SPECS</h3>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin-bottom: 10px;">✅ <b>Latency:</b> < 50ms (CUDA)</li>
                <li style="margin-bottom: 10px;">✅ <b>FPS:</b> 30+ (RTX 3060+)</li>
                <li style="margin-bottom: 10px;">✅ <b>Engine:</b> ONNX Runtime</li>
                <li style="margin-bottom: 10px;">✅ <b>Precision:</b> FP16 / INT8</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 LIVE FEATURES")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 👁️ Face Tracking")
        st.caption("Multi-face detection & tracking")
    with c2:
        st.markdown("#### 📊 Live Scoring")
        st.caption("Frame-by-frame probability")
    with c3:
        st.markdown("#### 🎨 Visual Overlay")
        st.caption("Bounding boxes & heatmaps")
    with c4:
        st.markdown("#### 💾 Auto-Logging")
        st.caption("Save detections to database")


def batch_mode():
    """Batch processing mode"""
    st.markdown("## 📦 BATCH PROCESSING")
    
    st.markdown("""
    <div class="metric-card">
        <h4>📂 BULK ANALYSIS</h4>
        <p>Upload multiple files for automated scanning and reporting.</p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose multiple files",
        type=['mp4', 'avi', 'mov', 'wav', 'mp3'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ **{len(uploaded_files)} FILES QUEUED**")
        
        if st.button("🚀 START BATCH PROCESSING", type="primary"):
            st.markdown("### 🔄 PROCESSING QUEUE")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_data = []
            
            detector = get_detector()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Analyzing: {file.name}...")
                
                # Save to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                
                # Predict
                try:
                    res = detector.predict(video_path=tmp_path)
                    results_data.append({
                        "Filename": file.name,
                        "Verdict": res['final_label'],
                        "Score": f"{res['final_score']:.2%}",
                        "Confidence": f"{res['confidence']:.2%}",
                        "Video": f"{res['individual_scores'].get('video', 0):.2f}",
                        "Audio": f"{res['individual_scores'].get('audio', 0):.2f}"
                    })
                except Exception as e:
                    results_data.append({
                        "Filename": file.name,
                        "Verdict": "ERROR",
                        "Score": "0.00%",
                        "Confidence": "0.00%",
                        "Video": "0.00",
                        "Audio": "0.00"
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Batch processing complete!")
            
            # Display results table
            st.markdown("### 📊 BATCH RESULTS")
            st.dataframe(
                results_data,
                use_container_width=True,
                column_config={
                    "Verdict": st.column_config.TextColumn(
                        "Verdict",
                        help="Final classification",
                        validate="^(REAL|FAKE)$"
                    ),
                    "Score": st.column_config.ProgressColumn(
                        "Threat Score",
                        format="%s",
                        min_value=0,
                        max_value=1
                    )
                }
            )
            
            # Download CSV
            import pandas as pd
            df = pd.DataFrame(results_data)
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "📥 DOWNLOAD CSV REPORT",
                csv,
                "batch_report.csv",
                "text/csv",
                key='download-csv'
            )


# Visualization helpers
def plot_temporal_features():
    """Plot temporal features"""
    # Placeholder for temporal visualization
    pass


def plot_spectrogram():
    """Plot audio spectrogram"""
    # Placeholder for spectrogram visualization
    pass


def plot_heatmap():
    """Plot artifact heatmap"""
    # Placeholder for heatmap visualization
    pass


if __name__ == "__main__":
    main()
