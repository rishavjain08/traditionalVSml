"""
Interactive Demo Section
Allows users to interact with both Traditional and ML models
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.traditional_model import predict_traditional


def render(model_data):
    """Render the Interactive Demo tab content"""
    st.markdown("### 🎮 Try Both Approaches")
    st.markdown("Configure match parameters and see predictions from both models")
    
    # Check if model is available
    if model_data is None:
        st.warning("⚠️ **No trained model available.** Please train a model using the 'Train Model' button in the sidebar.")
        st.info("You can still use the Traditional Programming approach below.")
    
    # Input controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        team1 = st.selectbox("Team 1", 
            ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
             'Kolkata Knight Riders', 'Delhi Capitals'])
        team1_form = st.slider("Team 1 Recent Form (wins in last 5)", 0, 5, 3)
    
    with col2:
        team2 = st.selectbox("Team 2",
            ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 
             'Kolkata Knight Riders', 'Delhi Capitals'])
        team2_form = st.slider("Team 2 Recent Form (wins in last 5)", 0, 5, 2)
    
    with col3:
        venue = st.selectbox("Venue",
            ['Wankhede Stadium', 'MA Chidambaram Stadium', 'Eden Gardens', 
             'M. Chinnaswamy Stadium', 'Arun Jaitley Stadium'])
        toss_winner = st.radio("Toss Winner", [team1, team2])
    
    # Make predictions
    if st.button("🎯 Predict Winner", type="primary"):
        # Traditional prediction
        trad_winner, trad_conf = predict_traditional(team1, team2, venue, toss_winner, 
                                                     team1_form, team2_form)
        
        # ML prediction (only if model exists)
        ml_winner = None
        ml_conf = None
        ml_prob = None
        
        if model_data is not None:
            model = model_data['model']
            
            # Prepare features for ML
            home_adv = 1 if ('Mumbai' in team1 and venue == 'Wankhede Stadium') else 0
            features = np.array([[
                team1_form, team2_form, home_adv,
                1 if toss_winner == team1 else 0,
                1,  # Assuming bat first
                hash(team1) % 10,
                hash(team2) % 10,
                hash(venue) % 10
            ]])
            
            ml_pred = model.predict(features)[0]
            ml_prob = model.predict_proba(features)[0]
            ml_winner = team1 if ml_pred == 1 else team2
            ml_conf = max(ml_prob)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🏆 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Traditional Programming")
            st.metric("Winner", trad_winner)
            st.metric("Confidence", f"{trad_conf:.1%}")
            
            if trad_conf < 0.6:
                st.warning("⚠️ Low confidence - mostly guessing!")
            
            # Show why
            st.info(f"""
            **Logic Used:**
            - Home advantage check: {'Yes' if venue == 'Wankhede Stadium' and team1 == 'Mumbai Indians' else 'No'}
            - Form-based rule: {'Yes' if abs(team1_form - team2_form) >= 2 else 'No'}
            - Toss advantage: {toss_winner}
            """)
        
        with col2:
            st.markdown("#### Machine Learning")
            if model_data is None:
                st.error("❌ Model not trained. Please train a model first.")
            else:
                st.metric("Winner", ml_winner)
                st.metric("Confidence", f"{ml_conf:.1%}")
                
                if ml_conf > 0.7:
                    st.success("✅ High confidence prediction!")
                
                # Feature importance
                home_adv = 1 if ('Mumbai' in team1 and venue == 'Wankhede Stadium') else 0
                st.info(f"""
                **Factors Considered:**
                - Recent form difference: {team1_form - team2_form}
                - Home advantage: {'Yes' if home_adv else 'No'}
                - Toss advantage: {'Yes' if toss_winner == team1 else 'No'}
                - Historical venue performance
                - Head-to-head records
                """)
        
        # Probability distribution (only if ML model is available)
        if model_data is not None and ml_prob is not None:
            st.markdown("---")
            st.markdown("### 📊 Win Probability Distribution")
            
            prob_data = pd.DataFrame({
                'Team': [team1, team2],
                'Traditional': [trad_conf if trad_winner == team1 else 1-trad_conf,
                              trad_conf if trad_winner == team2 else 1-trad_conf],
                'ML Model': [ml_prob[1], ml_prob[0]]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Traditional', x=prob_data['Team'], y=prob_data['Traditional'],
                                marker_color='lightblue'))
            fig.add_trace(go.Bar(name='ML Model', x=prob_data['Team'], y=prob_data['ML Model'],
                                marker_color='darkblue'))
            fig.update_layout(barmode='group', yaxis_title='Win Probability',
                             title='Prediction Comparison')
            st.plotly_chart(fig, use_container_width=True)

