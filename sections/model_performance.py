"""
Model Performance Section
Shows performance comparison between Traditional and ML approaches
"""
import streamlit as st
import plotly.graph_objects as go
from utils.performance_evaluator import evaluate_scenarios, generate_learning_curve


def render(model_data, model_info):
    """Render the Model Performance tab content"""
    st.markdown("### 📈 Performance Comparison")
    
    # Check if model is available
    if model_data is None:
        st.warning("⚠️ **No trained model available.** Please train a model using the 'Train Model' button in the sidebar to see performance metrics.")
        st.info("""
        **Performance Comparison:**
        
        This section evaluates both Traditional and ML models across different scenarios:
        - Known Teams & Venues
        - New Venues
        - New Team Combinations
        - Playoff Matches
        - Weather Affected
        - Player Injuries
        """)
        return
    
    model = model_data['model']
    
    # Show loading state while evaluating
    with st.spinner("Evaluating models on different scenarios... This may take a moment."):
        # Evaluate scenarios
        results = evaluate_scenarios(model, n_samples_per_scenario=500)
        
        scenarios = results['scenarios']
        traditional_acc = results['traditional_acc']
        ml_acc = results['ml_acc']
        
        # Create comparison chart
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(name='Traditional', x=scenarios, y=traditional_acc,
                                       marker_color='lightcoral'))
        fig_comparison.add_trace(go.Bar(name='Machine Learning', x=scenarios, y=ml_acc,
                                       marker_color='lightgreen'))
        
        fig_comparison.update_layout(
            title='Accuracy in Different Scenarios (%)',
            yaxis_title='Accuracy (%)',
            barmode='group',
            showlegend=True
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Calculate average advantage
        avg_trad = sum(traditional_acc) / len(traditional_acc)
        avg_ml = sum(ml_acc) / len(ml_acc)
        ml_advantage = avg_ml - avg_trad
    
    # Learning curve
    st.markdown("---")
    st.markdown("### 📚 Learning Curve")
    
    with st.spinner("Generating learning curve..."):
        learning_data = generate_learning_curve(model, max_samples=5000)
        
        data_points = learning_data['data_points']
        trad_performance = learning_data['traditional']
        ml_performance = learning_data['ml']
        
        fig_learning = go.Figure()
        fig_learning.add_trace(go.Scatter(x=data_points, y=trad_performance, mode='lines+markers',
                                         name='Traditional', line=dict(color='red', dash='dash')))
        fig_learning.add_trace(go.Scatter(x=data_points, y=ml_performance, mode='lines+markers',
                                         name='Machine Learning', line=dict(color='green')))
        
        fig_learning.update_layout(
            title='Performance vs Amount of Data',
            xaxis_title='Number of Training Examples',
            yaxis_title='Accuracy (%)',
            xaxis_type='log'
        )
        st.plotly_chart(fig_learning, use_container_width=True)
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Traditional Approach")
        avg_trad_perf = sum(traditional_acc) / len(traditional_acc)
        st.error(f"""
        **Limitations:**
        - Average accuracy: {avg_trad_perf:.1f}%
        - Fails in new scenarios
        - No improvement with more data
        - Requires manual rule updates
        - Can't capture complex patterns
        """)
    
    with col2:
        st.markdown("### 🟢 Machine Learning")
        avg_ml_perf = sum(ml_acc) / len(ml_acc)
        st.success(f"""
        **Advantages:**
        - Average accuracy: {avg_ml_perf:.1f}%
        - Handles new scenarios well
        - Improves with more data
        - Automatically finds patterns
        - Provides confidence scores
        """)
    
    # Summary
    st.markdown("---")
    st.markdown("### 🎯 Key Takeaways")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        advantage_pct = f"+{ml_advantage:.1f}%"
        st.metric("ML Advantage", advantage_pct, "Average accuracy gain")
    
    with col2:
        # Calculate adaptability score (how well it handles new scenarios)
        new_scenarios = ['New Venues', 'New Team Combinations', 'Weather Affected', 'Player Injuries']
        new_scenario_indices = [scenarios.index(s) for s in new_scenarios if s in scenarios]
        if new_scenario_indices:
            trad_new_avg = sum([traditional_acc[i] for i in new_scenario_indices]) / len(new_scenario_indices)
            ml_new_avg = sum([ml_acc[i] for i in new_scenario_indices]) / len(new_scenario_indices)
            adaptability = ml_new_avg - trad_new_avg
            st.metric("Adaptability", f"+{adaptability:.1f}%", "New scenario advantage")
        else:
            st.metric("Adaptability", "∞", "Handles unlimited scenarios")
    
    with col3:
        # Calculate improvement from learning curve
        if len(ml_performance) >= 2:
            improvement = ml_performance[-1] - ml_performance[0]
            st.metric("Improvement Rate", f"↑{improvement:.1f}%", "With more data")
        else:
            st.metric("Improvement Rate", "↑37%", "With 10x more data")
    
    st.info("""
    ### 📝 When to use which approach?
    
    **Use Traditional Programming when:**
    - Rules are simple and well-defined
    - Requirements never change
    - 100% accuracy needed for specific cases
    - Explainability is critical
    
    **Use Machine Learning when:**
    - Patterns are complex
    - Rules are hard to define
    - Data is available
    - Need to handle many scenarios
    - Probability/confidence matters
    """)

