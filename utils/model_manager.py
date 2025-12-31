"""
Model Manager - Handles loading and saving of trained models
"""
import pickle
import os
from typing import Optional, Dict, Any


MODEL_PATH = 'model.pkl'


def model_exists(model_path: str = MODEL_PATH) -> bool:
    """Check if a trained model exists"""
    return os.path.exists(model_path)


def load_model(model_path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    """
    Load model from pickle file
    
    Returns:
        dict with keys: 'model', 'train_acc', 'test_acc', 'n_samples', 'feature_names'
        None if model doesn't exist
    """
    if not model_exists(model_path):
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_model_info(model_path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    """
    Get model information without loading the full model
    
    Returns:
        dict with model metadata or None
    """
    model_data = load_model(model_path)
    if model_data is None:
        return None
    
    return {
        'train_acc': model_data.get('train_acc'),
        'test_acc': model_data.get('test_acc'),
        'n_samples': model_data.get('n_samples'),
        'feature_names': model_data.get('feature_names', [])
    }


def delete_model(model_path: str = MODEL_PATH) -> bool:
    """Delete the model file"""
    try:
        if model_exists(model_path):
            os.remove(model_path)
            return True
        return False
    except Exception as e:
        print(f"Error deleting model: {e}")
        return False

