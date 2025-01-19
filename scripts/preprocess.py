import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """
    Preprocess the Spotify dataset to handle missing values, scale numerical features,
    encode categorical data, and extract useful datetime components.
    
    Parameters:
        df (pd.DataFrame): Input dataset to preprocess.

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset.
    """
    # Step 1: Print initial dataset details
    print("=== Dataset Information Before Processing ===")
    print(df.info(), "\n")
    print("=== Dataset Summary Statistics Before Processing ===")
    print(df.describe(), "\n")
    
    # Step 2: Check for missing values
    print("=== Missing Values Before Processing ===")
    missing_values_before = df.isnull().sum()
    print(missing_values_before[missing_values_before > 0], "\n")

    # Step 3: Handle missing values
    print("Handling missing values...")
    df['track_name'].fillna('Unknown', inplace=True)
    df['track_artist'].fillna('Unknown', inplace=True)
    df['track_album_name'].fillna('Unknown', inplace=True)
    
    numerical_columns = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

    # Step 4: Encode categorical columns
    print("Encoding categorical columns...")
    label_encoder = LabelEncoder()
    if df['playlist_genre'].dtype == 'object':
        df['playlist_genre'] = label_encoder.fit_transform(df['playlist_genre'])
    if df['playlist_subgenre'].dtype == 'object':
        df['playlist_subgenre'] = label_encoder.fit_transform(df['playlist_subgenre'])

    # Step 5: Scale numerical features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Step 6: Extract datetime components
    print("Processing datetime columns...")
    df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')
    df['release_year'] = df['track_album_release_date'].dt.year

    # Step 7: Validate the processed data
    print("\n=== Dataset Information After Processing ===")
    print(df.info(), "\n")
    print("=== Dataset Summary Statistics After Processing ===")
    print(df.describe(), "\n")
    
    # Step 8: Check for remaining missing values
    print("=== Missing Values After Processing ===")
    missing_values_after = df.isnull().sum()
    print(missing_values_after[missing_values_after > 0])

    return df

if __name__ == "__main__":
    # Define dataset path
    data_path = 'data/spotify_dataset.csv'

    # Step 1: Load the dataset
    print(f"Loading dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        print("Dataset successfully loaded.\n")
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found. Ensure the path is correct.")
        exit(1)

    # Step 2: Preprocess the data
    print("Starting preprocessing...")
    df_cleaned = preprocess_data(df)

    # Step 3: Save the cleaned dataset
    output_path = 'data/preprocessed_data.csv'
    print(f"\nSaving preprocessed dataset to {output_path}...")
    df_cleaned.to_csv(output_path, index=False)
    print("Preprocessed dataset successfully saved.")
