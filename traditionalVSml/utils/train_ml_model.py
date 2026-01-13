import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from traditionalVSml.utils.data_generator import generate_ipl_data_raw


def train_and_save_model(n_samples=2000, model_path='model.pkl'):
    """
    Generate dataset, train model, and save as pickle file.
    
    Args:
        n_samples: Number of samples to generate for training
        model_path: Path to save the pickle file
        
    Returns:
        dict: Contains train_acc, test_acc, n_samples, and model_path
    """
    # Generate dataset
    df = generate_ipl_data_raw(n_samples)
    
    # Extract features
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Calculate accuracies
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Save model to pickle file
    model_data = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_samples': n_samples,
        'feature_names': ['Team1 Form', 'Team2 Form', 'Home Advantage', 'Toss Winner', 
                         'Toss Decision', 'Team1 Strength', 'Team2 Strength', 'Venue Factor']
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_samples': n_samples,
        'model_path': model_path,
        'success': True
    }


def train_ml_model_raw(df):
    """
    Legacy function for backward compatibility.
    This function is kept for any existing code that might use it.
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, train_acc, test_acc
