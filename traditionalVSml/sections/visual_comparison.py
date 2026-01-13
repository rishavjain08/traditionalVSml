"""
Visual Comparison Section
Shows side-by-side comparison of Traditional Programming vs Machine Learning
"""
import streamlit as st


def render():
    """Render the Visual Comparison tab content"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Traditional Programming")
        st.code("""
def predict_winner(team1, team2, venue):
    # Fixed rules written by programmer
    if venue == "Wankhede" and team1 == "MI":
        return "Mumbai Indians"  # Rule 1
    elif venue == "Chepauk" and team2 == "CSK":
        return "Chennai Super Kings"  # Rule 2
    else:
        return "Toss Winner"  # Default rule
        
# Problems:
# ❌ Only works for specific scenarios
# ❌ Can't handle new situations
# ❌ No confidence score
# ❌ Doesn't improve over time
        """, language='python')
        
        # Pros and Cons
        st.success("✅ **Pros:**")
        st.markdown("""
        - Simple to understand
        - Fast execution
        - Predictable behavior
        - No training needed
        """)
        
        st.error("❌ **Cons:**")
        st.markdown("""
        - Limited to known scenarios
        - Can't adapt to new patterns
        - Manual rule updates needed
        - No probability/confidence
        """)
    
    with col2:
        st.markdown("### 🤖 Machine Learning")
        st.code("""
# Train model on historical data
model = LinearRegression()
model.fit(X_train, y_train)  # Learn patterns

def predict_winner(features):
    # Model learns patterns from data
    prediction = model.predict(features)
    confidence = model.predict_proba(features)
    
    return prediction, confidence
    
# Advantages:
# ✅ Handles ANY scenario
# ✅ Provides confidence scores
# ✅ Improves with more data
# ✅ Finds complex patterns
        """, language='python')
        
        st.success("✅ **Pros:**")
        st.markdown("""
        - Learns from data automatically
        - Handles complex patterns
        - Provides confidence scores
        - Improves with more data
        """)
        
        st.warning("⚠️ **Cons:**")
        st.markdown("""
        - Needs training data
        - "Black box" nature
        - Requires more resources
        - Can overfit if not careful
        """)
    
    # Flowchart comparison
    st.markdown("---")
    st.markdown("### 🔄 Logic Flow Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Traditional Flow:**")
        st.markdown("""
        ```mermaid
        graph TD
            A[Input: Teams, Venue] --> B{Is MI at Wankhede?}
            B -->|Yes| C[Predict: MI Wins]
            B -->|No| D{Is CSK at Chepauk?}
            D -->|Yes| E[Predict: CSK Wins]
            D -->|No| F[Predict: Random]
        ```
        """)
        st.info("💡 **Fixed decision tree with hardcoded rules**")
    
    with col2:
        st.markdown("**ML Flow:**")
        st.markdown("""
        ```mermaid
        graph TD
            A[Historical Data] --> B[Feature Extraction]
            B --> C[Train Model]
            C --> D[Learned Patterns]
            E[New Match Input] --> F[Apply Model]
            D --> F
            F --> G[Prediction + Confidence]
        ```
        """)
        st.info("💡 **Dynamic patterns learned from data**")

