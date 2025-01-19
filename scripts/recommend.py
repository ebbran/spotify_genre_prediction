import pandas as pd
import joblib
import random

# Genre mapping from cluster number to genre name
genre_mapping = {
    0: 'pop',
    1: 'rap',
    2: 'rock',
    3: 'latin',
    4: 'r&b',
    5: 'edm'
}

def recommend_songs(clustered_data, playlist_id, model, num_recommendations=10):
    """
    Recommends songs based on the cluster of a given playlist using a trained KMeans model.

    Parameters:
    - clustered_data (DataFrame): The dataset with clustered songs.
    - playlist_id (str): The playlist ID for which recommendations are needed.
    - model (KMeans): The trained KMeans model.
    - num_recommendations (int): Number of recommendations to return.

    Returns:
    - DataFrame: A sample of recommended songs from the same cluster as the given playlist.
    """
    # Ensure num_recommendations is an integer
    if isinstance(num_recommendations, str):
        try:
            num_recommendations = int(num_recommendations)
        except ValueError:
            raise ValueError(f"Invalid num_recommendations value: {num_recommendations}")

    # Ensure the model has the 'predict' method (valid KMeans model)
    if not hasattr(model, 'predict'):
        raise ValueError("Provided model is not valid for prediction.")

    # Get the features for the playlist based on playlist_id
    features = get_features(clustered_data, playlist_id)
    if features is None:
        raise ValueError(f"Playlist ID {playlist_id} not found in the dataset.")

    # Predict the cluster for the playlist
    cluster_id = model.predict([features])[0]

    # Retrieve songs that belong to the same cluster as the given playlist
    cluster_songs = clustered_data[clustered_data['cluster'] == cluster_id]

    # If no songs found in the predicted cluster, raise an error
    if cluster_songs.empty:
        raise ValueError(f"No songs found in cluster {cluster_id} for playlist {playlist_id}.")

    # Map the cluster ID to the genre name
    cluster_songs['genre'] = cluster_songs['cluster'].map(genre_mapping)

    # Return a random sample of songs, up to num_recommendations
    return cluster_songs[['track_name', 'track_artist', 'genre']].sample(min(num_recommendations, len(cluster_songs)))


def get_features(df, playlist_id):
    """
    Extracts the feature values of a given playlist based on its ID and prepares it for prediction.

    Parameters:
    - df (DataFrame): The dataset containing playlist features.
    - playlist_id (str): The playlist ID whose features are to be extracted.

    Returns:
    - list: Feature values for the playlist, including both numerical and categorical features.
    """
    # Columns required for feature extraction
    numerical_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 
                         'tempo', 'duration_ms']
    
    categorical_columns = ['playlist_genre', 'playlist_subgenre']

    # Check if the dataset contains all the required numerical columns
    if not set(numerical_columns).issubset(df.columns):
        raise ValueError(f"Dataset does not contain all required numerical feature columns. Missing: {set(numerical_columns) - set(df.columns)}")
    
    # Extract playlist data for the given playlist_id
    playlist_data = df[df['playlist_id'] == playlist_id]
    if playlist_data.empty:
        return None  # Playlist ID not found
    
    # Extract numerical features
    numerical_features = playlist_data[numerical_columns].mean().tolist()

    # One-hot encode the categorical features
    categorical_features = pd.get_dummies(playlist_data[categorical_columns], drop_first=True).mean().tolist()

    # Return combined numerical and categorical features
    return numerical_features + categorical_features
