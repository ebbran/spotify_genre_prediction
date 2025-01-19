import pandas as pd 
import joblib
import os
from jinja2 import Template
import webbrowser

# Import all necessary functions from the given modules
from preprocess import preprocess_data
from analysis import analyze_data
from cluster import perform_clustering
from train import train_model
from recommend import recommend_songs

def generate_html_report(preprocessed_data, analysis_images, clustering_images, recommendations, output_path="report.html"):
    """
    Generate an HTML report with analysis visualizations, clustering results, and song recommendations.

    Parameters:
        preprocessed_data (pd.DataFrame): The cleaned and preprocessed dataset.
        analysis_images (list): List of file paths for analysis visualizations.
        clustering_images (list): List of file paths for clustering visualizations.
        recommendations (pd.DataFrame): Recommended songs with their details.
        output_path (str): Path to save the HTML report.
    """
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spotify Data Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            img { max-width: 100%; height: auto; margin-bottom: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f4f4f4; }
        </style>
    </head>
    <body>
        <h1>Spotify Data Analysis and Recommendations Report</h1>

        <h2>Analysis Visualizations</h2>
        {% for img in analysis_images %}
            <img src="{{ img }}" alt="Analysis Visualization">
        {% endfor %}

        <h2>Clustering Visualizations</h2>
        {% for img in clustering_images %}
            <img src="{{ img }}" alt="Clustering Visualization">
        {% endfor %}

        <h2>Recommended Songs</h2>
        <table>
            <tr>
                <th>Track Name</th>
                <th>Artist</th>
                <th>Genre</th>
            </tr>
            {% for _, row in recommendations.iterrows() %}
            <tr>
                <td>{{ row['track_name'] }}</td>
                <td>{{ row['track_artist'] }}</td>
                <td>{{ row['genre'] }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """

    # Render the HTML with the template
    html_content = Template(template).render(
        analysis_images=analysis_images,
        clustering_images=clustering_images,
        recommendations=recommendations
    )

    # Save to file
    with open(output_path, "w") as file:
        file.write(html_content)
    print(f"HTML report saved to {output_path}")

    # Automatically open the report in a browser
    webbrowser.open(f"file://{os.path.abspath(output_path)}")

def main():
    # File paths
    data_path = "data/spotify_dataset.csv"
    preprocessed_data_path = "data/preprocessed_data.csv"
    clustering_data_path = "data/clustered_data.csv"
    kmeans_model_path = "kmeans_model.pkl"
    report_path = "report.html"

    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    raw_data = pd.read_csv(data_path)
    preprocessed_data = preprocess_data(raw_data)
    preprocessed_data.to_csv(preprocessed_data_path, index=False)

    # Step 2: Analyze data and generate visualizations
    print("Analyzing data and generating visualizations...")
    analysis_images, _ = analyze_data(preprocessed_data, output_folder="figures")

    # Step 3: Perform clustering
    print("Performing clustering...")
    clustered_data, kmeans_model, clustering_images, _ = perform_clustering(preprocessed_data, output_dir="clustering_images")
    clustered_data.to_csv(clustering_data_path, index=False)

    # Step 4: Train or load the KMeans model
    print("Training or loading the KMeans model...")
    trained_model = train_model(preprocessed_data, n_clusters=5, model_path=kmeans_model_path)

    # Step 5: Generate recommendations
    print("Generating recommendations...")

    # Get a valid playlist ID from the clustered data (ensure it's a real ID from the dataset)
    playlist_ids = clustered_data['playlist_id'].unique()
    playlist_id = playlist_ids[0]  # You can choose any playlist_id, or loop through multiple

    recommendations = recommend_songs(clustered_data, playlist_id=playlist_id, model=trained_model, num_recommendations=10)

    # Step 6: Generate the HTML report
    print("Generating the HTML report...")
    generate_html_report(
        preprocessed_data=preprocessed_data,
        analysis_images=analysis_images,
        clustering_images=clustering_images,
        recommendations=recommendations,
        output_path=report_path
    )

if __name__ == "__main__":
    main()
