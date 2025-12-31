"""
Common utilities and helper functions
"""
import streamlit as st


def get_custom_css():
    """Returns custom CSS for the app"""
    return """
    <style>
        .main > div {
            padding-top: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.2);
            padding: 5% 5% 5% 10%;
            border-radius: 10px;
        }
    </style>
    """


def render_sidebar():
    """Render the sidebar content"""
    st.sidebar.header("📚 About This App")
    st.sidebar.info(
        "This interactive demo shows how Machine Learning differs from traditional programming "
        "using IPL match predictions as an example."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Key Learning Points")
    st.sidebar.markdown("""
    1. **Traditional**: Fixed rules, limited scenarios
    2. **ML**: Learns from data, handles any scenario
    3. **ML provides confidence scores**
    4. **ML improves with more data**
    """)


def render_footer():
    """Render the footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Created by Rishav Jain | Rule Based Programming vs Machine Learning</p>
        <p>🏏 Cricket + 🤖 ML = Better Predictions!</p>
    </div>
    """, unsafe_allow_html=True)

