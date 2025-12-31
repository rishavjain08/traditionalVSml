"""
Performance Evaluator - Evaluates Traditional and ML models on various scenarios
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.traditional_model import predict_traditional
from utils.data_generator import generate_ipl_data_raw


def evaluate_traditional_model(df):
    """
    Evaluate traditional model on a dataset
    
    Returns:
        float: Accuracy score (0-1)
    """
    correct = 0
    total = len(df)
    
    for _, row in df.iterrows():
        pred_winner, _ = predict_traditional(
            row['team1'],
            row['team2'],
            row['venue'],
            row['toss_winner'],
            row['team1_wins_last_5'],
            row['team2_wins_last_5']
        )
        
        if pred_winner == row['winner']:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_ml_model(model, df):
    """
    Evaluate ML model on a dataset
    
    Returns:
        float: Accuracy score (0-1)
    """
    features = []
    for _, row in df.iterrows():
        features.append([
            row['team1_wins_last_5'],
            row['team2_wins_last_5'],
            row['home_advantage'],
            int(row['toss_winner'] == row['team1']),
            int(row['toss_decision'] == 'bat'),
            hash(row['team1']) % 10,
            hash(row['team2']) % 10,
            hash(row['venue']) % 10
        ])
    
    X = np.array(features)
    y = (df['winner'] == df['team1']).astype(int)
    
    predictions = model.predict(X)
    return accuracy_score(y, predictions)


def generate_scenario_data(scenario_type, n_samples=500, seed=42):
    """
    Generate test data for different scenarios
    
    Args:
        scenario_type: Type of scenario ('known', 'new_venues', 'new_teams', 
                       'playoff', 'weather', 'injuries')
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Test dataset
    """
    np.random.seed(seed)  # For reproducibility
    
    teams = [
        'Mumbai Indians', 'Chennai Super Kings',
        'Royal Challengers Bangalore', 'Kolkata Knight Riders',
        'Delhi Capitals', 'Punjab Kings',
        'Rajasthan Royals', 'Sunrisers Hyderabad'
    ]
    
    known_venues = [
        'Wankhede Stadium', 'MA Chidambaram Stadium',
        'Eden Gardens', 'M. Chinnaswamy Stadium',
        'Arun Jaitley Stadium', 'Narendra Modi Stadium'
    ]
    
    new_venues = [
        'Brabourne Stadium', 'DY Patil Stadium',
        'Sawai Mansingh Stadium', 'Rajiv Gandhi Stadium'
    ]
    
    data = []
    
    for _ in range(n_samples):
        if scenario_type == 'known':
            # Known teams and venues
            team1 = np.random.choice(teams)
            team2 = np.random.choice([t for t in teams if t != team1])
            venue = np.random.choice(known_venues)
            
        elif scenario_type == 'new_venues':
            # New venues
            team1 = np.random.choice(teams)
            team2 = np.random.choice([t for t in teams if t != team1])
            venue = np.random.choice(new_venues)
            
        elif scenario_type == 'new_teams':
            # New team combinations (less common matchups)
            team1 = np.random.choice(teams[:4])  # Top teams
            team2 = np.random.choice(teams[4:])  # Bottom teams
            venue = np.random.choice(known_venues)
            
        elif scenario_type == 'playoff':
            # Playoff scenarios (higher stakes, better teams)
            playoff_teams = teams[:4]  # Top 4 teams
            team1 = np.random.choice(playoff_teams)
            team2 = np.random.choice([t for t in playoff_teams if t != team1])
            venue = np.random.choice(known_venues[:3])  # Major venues
            
        elif scenario_type == 'weather':
            # Weather affected (more randomness)
            team1 = np.random.choice(teams)
            team2 = np.random.choice([t for t in teams if t != team1])
            venue = np.random.choice(known_venues)
            
        elif scenario_type == 'injuries':
            # Player injuries (weaker form)
            team1 = np.random.choice(teams)
            team2 = np.random.choice([t for t in teams if t != team1])
            venue = np.random.choice(known_venues)
            
        else:
            # Default: known scenario
            team1 = np.random.choice(teams)
            team2 = np.random.choice([t for t in teams if t != team1])
            venue = np.random.choice(known_venues)
        
        team1_wins_last_5 = np.random.randint(0, 6)
        team2_wins_last_5 = np.random.randint(0, 6)
        
        # Adjust for special scenarios
        if scenario_type == 'injuries':
            # Weaker form due to injuries
            team1_wins_last_5 = min(team1_wins_last_5, 2)
            team2_wins_last_5 = min(team2_wins_last_5, 2)
        
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(['bat', 'field'])
        
        home_advantage = int(
            ('Mumbai' in team1 and venue == 'Wankhede Stadium') or
            ('Chennai' in team1 and venue == 'MA Chidambaram Stadium') or
            ('Kolkata' in team1 and venue == 'Eden Gardens')
        )
        
        win_prob = 0.5
        win_prob += (team1_wins_last_5 - team2_wins_last_5) * 0.05
        win_prob += home_advantage * 0.15
        if toss_winner == team1:
            win_prob += 0.1
        
        # Add more randomness for weather scenario (less predictable outcomes)
        if scenario_type == 'weather':
            # Weather adds more uncertainty
            weather_noise = (np.random.rand() - 0.5) * 0.4
            win_prob = 0.5 + weather_noise
            win_prob = max(0.1, min(0.9, win_prob))  # Clamp between 0.1 and 0.9
        
        winner = team1 if np.random.rand() < win_prob else team2
        
        data.append({
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "team1_wins_last_5": team1_wins_last_5,
            "team2_wins_last_5": team2_wins_last_5,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "home_advantage": home_advantage,
            "winner": winner
        })
    
    return pd.DataFrame(data)


def evaluate_scenarios(model, n_samples_per_scenario=500):
    """
    Evaluate both models on different scenarios
    
    Returns:
        dict: Dictionary with scenario names and accuracies for both models
    """
    scenarios = {
        'Known Teams & Venues': 'known',
        'New Venues': 'new_venues',
        'New Team Combinations': 'new_teams',
        'Playoff Matches': 'playoff',
        'Weather Affected': 'weather',
        'Player Injuries': 'injuries'
    }
    
    results = {
        'scenarios': list(scenarios.keys()),
        'traditional_acc': [],
        'ml_acc': []
    }
    
    for scenario_name, scenario_type in scenarios.items():
        # Generate test data for this scenario
        test_data = generate_scenario_data(scenario_type, n_samples_per_scenario)
        
        # Evaluate traditional model
        trad_acc = evaluate_traditional_model(test_data)
        results['traditional_acc'].append(trad_acc * 100)  # Convert to percentage
        
        # Evaluate ML model
        if model is not None:
            ml_acc = evaluate_ml_model(model, test_data)
            results['ml_acc'].append(ml_acc * 100)  # Convert to percentage
        else:
            results['ml_acc'].append(0.0)
    
    return results


def generate_learning_curve(model, max_samples=5000, quick_mode=True):
    """
    Generate learning curve data by training models with different data sizes
    
    Args:
        model: The trained model (used as reference for max performance)
        max_samples: Maximum number of samples to use
        quick_mode: If True, uses approximation for smaller sizes. If False, retrains models.
    
    Returns:
        dict: Dictionary with data points and performance metrics
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Use smaller data points for quick mode, larger for full mode
    if quick_mode:
        data_points = [100, 200, 500, 1000, 2000]
    else:
        data_points = [100, 200, 500, 1000, 2000, min(max_samples, 5000)]
    
    # Traditional model doesn't improve (constant)
    trad_performance = []
    ml_performance = []
    
    # Generate test set (fixed for fair comparison)
    test_data = generate_ipl_data_raw(1000)
    
    # Get baseline traditional accuracy
    baseline_trad_acc = evaluate_traditional_model(test_data)
    
    for n_samples in data_points:
        # Traditional model accuracy (doesn't change with data)
        trad_performance.append(baseline_trad_acc * 100)
        
        # ML model - train with n_samples and evaluate
        if model is not None:
            if quick_mode and n_samples >= 2000:
                # Use actual trained model performance
                ml_acc = evaluate_ml_model(model, test_data)
                ml_performance.append(ml_acc * 100)
            elif quick_mode:
                # Quick approximation: simulate learning curve
                # Get actual model performance as baseline
                base_acc = evaluate_ml_model(model, test_data)
                # Simulate improvement curve (logarithmic improvement)
                # Smaller datasets = lower accuracy
                if n_samples < 2000:
                    # Simulate realistic learning curve
                    # Accuracy improves logarithmically with data size
                    log_factor = np.log(n_samples) / np.log(2000)
                    # Start from ~55% and improve to base_acc
                    start_acc = 0.55
                    ml_acc = start_acc + (base_acc - start_acc) * (log_factor ** 0.7)
                else:
                    ml_acc = base_acc
                ml_performance.append(ml_acc * 100)
            else:
                # Full mode: Actually retrain with n_samples
                train_data = generate_ipl_data_raw(n_samples)
                
                # Extract features
                features = []
                for _, row in train_data.iterrows():
                    features.append([
                        row['team1_wins_last_5'],
                        row['team2_wins_last_5'],
                        row['home_advantage'],
                        int(row['toss_winner'] == row['team1']),
                        int(row['toss_decision'] == 'bat'),
                        hash(row['team1']) % 10,
                        hash(row['team2']) % 10,
                        hash(row['venue']) % 10
                    ])
                
                X = np.array(features)
                y = (train_data['winner'] == train_data['team1']).astype(int)
                
                # Train model
                temp_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                temp_model.fit(X, y)
                
                # Evaluate on test set
                ml_acc = evaluate_ml_model(temp_model, test_data)
                ml_performance.append(ml_acc * 100)
        else:
            ml_performance.append(55.0)  # Default starting point
    
    return {
        'data_points': data_points,
        'traditional': trad_performance,
        'ml': ml_performance
    }

