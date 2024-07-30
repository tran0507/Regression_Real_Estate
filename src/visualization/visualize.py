import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()

def plot_feature_importance(feature_names, feature_importances):
    """
    Plot a bar chart showing the feature importances.
    
    Args:
        feature_names (list): List of feature names.
        feature_importances (list): List of feature importance values.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.title('Feature Importance', fontsize=16)
    plt.show()

