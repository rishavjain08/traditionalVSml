"""
How ML Works Section
Explains how machine learning models work and make predictions
"""
import streamlit as st
import plotly.express as px


def render(model_data, model_info):
    """Render the How ML Works tab content"""
    st.markdown("### 🧠 Understanding Machine Learning")
    
    # Check if model is available
    if model_data is None or model_info is None:
        st.warning("⚠️ **No trained model available.** Please train a model using the 'Train Model' button in the sidebar to see model details.")
        st.info("""
        **How ML Works:**
        
        **1. Training Phase:**
        - Analyzes thousands of historical matches
        - Identifies patterns in winning teams
        - Learns which factors matter most
        
        **2. Prediction Phase:**
        - Takes new match parameters
        - Applies learned patterns
        - Calculates probability for each team
        - Provides confidence score
        
        **3. Continuous Improvement:**
        - Gets feedback from actual results
        - Updates patterns with new data
        - Becomes more accurate over time
        """)
        return
    
    model = model_data['model']
    train_acc = model_info['train_acc']
    test_acc = model_info['test_acc']
    feature_names = model_info['feature_names']
    
    # Feature importance
    importance = model.feature_importances_
    
    # Create feature importance plot
    fig_importance = px.bar(
        x=importance, 
        y=feature_names, 
        orientation='h',
        title='Feature Importance in ML Model',
        labels={'x': 'Importance', 'y': 'Features'},
        color=importance,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Model Statistics")
        st.metric("Training Accuracy", f"{train_acc:.1%}")
        st.metric("Testing Accuracy", f"{test_acc:.1%}")
        st.metric("Number of Trees", "100")
        st.metric("Features Used", "8")
        st.metric("Training Samples", f"{model_info['n_samples']:,}")
        
        st.info("""
        **What the model learned:**
        - Recent form is most important
        - Home advantage matters significantly
        - Toss has moderate impact
        - Venue history influences outcomes
        """)
    
    with col2:
        st.markdown("### 🎯 How ML Makes Decisions")
        
        st.markdown("""
        **1. Training Phase:**
        - Analyzes thousands of historical matches
        - Identifies patterns in winning teams
        - Learns which factors matter most
        
        **2. Prediction Phase:**
        - Takes new match parameters
        - Applies learned patterns
        - Calculates probability for each team
        - Provides confidence score
        
        **3. Continuous Improvement:**
        - Gets feedback from actual results
        - Updates patterns with new data
        - Becomes more accurate over time
        """)
    
    # Show sample decision tree
    st.markdown("---")
    st.markdown("### 🌳 Sample Decision Path")
    st.code("""
    IF team1_form > 3.5:
        IF home_advantage == 1:
            Probability(team1_wins) = 0.78
        ELSE:
            IF toss_winner == team1:
                Probability(team1_wins) = 0.65
            ELSE:
                Probability(team1_wins) = 0.52
    ELSE:
        IF team2_form > 3.5:
            Probability(team1_wins) = 0.35
        ELSE:
            Probability(team1_wins) = 0.48
    """)
    
    st.success("💡 **Note:** ML model uses 100 such trees and averages their predictions!")

