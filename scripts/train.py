import pandas as pd
import joblib
import os
from sklearn.cluster import KMeans

def train_model(df, n_clusters, model_path='kmeans_model.pkl'):
    """
    Trains a KMeans model using the provided DataFrame and saves the trained model.
    If the model already exists, it will load the saved model.
    
    Parameters:
    - df (DataFrame): Preprocessed DataFrame with necessary features.
    - n_clusters (int): Number of clusters to use in KMeans.
    - model_path (str): Path to save or load the model.
    
    Returns:
    - trained_model: The trained or loaded KMeans model.
    """
    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        return joblib.load(model_path)

    # Validate the necessary columns before training
    numerical_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 
                         'tempo', 'duration_ms']
    categorical_columns = ['playlist_genre', 'playlist_subgenre']
    
    missing_columns = set(numerical_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing numerical columns: {missing_columns}")
    
    # Preprocess the features
    X_numerical = df[numerical_columns]
    X_categorical = pd.get_dummies(df[categorical_columns], drop_first=True)
    X = pd.concat([X_numerical, X_categorical], axis=1)

    # Train the KMeans model
    print(f"Training KMeans model with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)

    # Save the trained model
    joblib.dump(kmeans, model_path)
    print(f"Model saved to: {model_path}")

    return kmeans


# Example Usage
if __name__ == "__main__":
    # Load preprocessed dataset
    df = pd.read_csv('data/preprocessed_data.csv')

    # Train the model and save it
    trained_model = train_model(df, n_clusters=5, model_path='kmeans_model.pkl')
