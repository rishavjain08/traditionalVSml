import streamlit as st

from traditionalVSml.utils.train_ml_model import train_and_save_model
from traditionalVSml.utils.model_manager import load_model, model_exists, get_model_info
from traditionalVSml.utils.common import get_custom_css, render_footer
from traditionalVSml.sections.visual_comparison import render as render_visual_comparison
from traditionalVSml.sections.interactive_demo import render as render_interactive_demo
from traditionalVSml.sections.how_ml_works import render as render_how_ml_works
from traditionalVSml.sections.model_performance import render as render_model_performance

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="IPL Prediction: Traditional vs ML",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ MAIN PAGE DESIGN ------------------

# Custom CSS for better styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Title and description
st.markdown(
    """
    <h1 style="text-align: center;">🏏 IPL Winner Prediction</h1>
    <h3 style="text-align: center;">Traditional Programming vs Machine Learning</h3>
    """,
    unsafe_allow_html=True
)

# ------------------ MODEL MANAGEMENT ------------------

# Initialize session state for model
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
    st.session_state.model_info = None

# Load model if it exists and not already loaded
if st.session_state.model_data is None and model_exists():
    st.session_state.model_data = load_model()
    if st.session_state.model_data:
        st.session_state.model_info = get_model_info()

# Sidebar with model training
st.sidebar.header("📚 About This App")
st.sidebar.info(
    "This interactive demo shows how Machine Learning differs from traditional programming "
    "using IPL match predictions as an example."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Management")

# Model status
if st.session_state.model_data:
    st.sidebar.success("✅ Model Trained")
    if st.session_state.model_info:
        st.sidebar.metric("Training Accuracy", f"{st.session_state.model_info['train_acc']:.1%}")
        st.sidebar.metric("Test Accuracy", f"{st.session_state.model_info['test_acc']:.1%}")
        st.sidebar.metric("Training Samples", f"{st.session_state.model_info['n_samples']:,}")
else:
    st.sidebar.warning("⚠️ No Model Trained")
    st.sidebar.info("Click 'Train Model' to generate dataset and train the ML model.")

# Train Model button
st.sidebar.markdown("---")
n_samples = st.sidebar.number_input(
    "Training Samples", 
    min_value=500, 
    max_value=10000, 
    value=2000, 
    step=500,
    help="Number of samples to generate for training"
)

if st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True):
    with st.sidebar:
        with st.spinner("Training model... This may take a moment."):
            try:
                result = train_and_save_model(n_samples=n_samples)
                if result['success']:
                    # Reload model
                    st.session_state.model_data = load_model()
                    st.session_state.model_info = get_model_info()
                    st.sidebar.success(f"✅ Model trained successfully!")
                    st.sidebar.success(f"Training Accuracy: {result['train_acc']:.1%}")
                    st.sidebar.success(f"Test Accuracy: {result['test_acc']:.1%}")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error training model: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Key Learning Points")
st.sidebar.markdown("""
1. **Traditional**: Fixed rules, limited scenarios
2. **ML**: Learns from data, handles any scenario
3. **ML provides confidence scores**
4. **ML improves with more data**
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Comparison", "🎮 Interactive Demo", 
                                   "🧠 How ML Works", "📈 Model Performance"])

# Tab 1: Visual Comparison
with tab1:
    render_visual_comparison()

# Tab 2: Interactive Demo
with tab2:
    render_interactive_demo(st.session_state.model_data)

# Tab 3: How ML Works
with tab3:
    render_how_ml_works(st.session_state.model_data, st.session_state.model_info)

# Tab 4: Model Performance
with tab4:
    render_model_performance(st.session_state.model_data, st.session_state.model_info)

# Footer
render_footer()
