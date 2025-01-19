Spotify Data Analysis and Recommendation System
Overview
This project is a comprehensive data analysis and recommendation system for Spotify datasets. It processes raw Spotify data, performs exploratory data analysis (EDA), clusters songs based on features, and generates personalized song recommendations. The project generates an HTML report summarizing analysis results, visualizations, and recommendations.

Features
Data Preprocessing:

Cleans and processes raw Spotify datasets.
Handles missing values and encodes categorical features.
Scales numerical features for uniformity.
Extracts additional features like release year.
Exploratory Data Analysis (EDA):

Creates 10 detailed visualizations, including:
Correlation heatmaps.
Distribution plots for features.
Pair plots and box plots for deeper insights.
Scatter plots, KDE plots, and bar plots.
Clustering:

Groups songs into clusters based on their features using KMeans.
Visualizes clustering results (e.g., cluster distribution, scatter plots).
Evaluates clusters using silhouette and Calinski-Harabasz scores.
Recommendations:

Generates personalized song recommendations based on a playlist's cluster.
Recommends songs from the same cluster, ensuring genre similarity.
HTML Report Generation:

Summarizes data preprocessing, analysis, clustering, and recommendations.
Includes interactive visualizations and a table of recommended songs.
Automatically opens the report in the browser after completion.
Installation and Setup

Create a Virtual Environment:

bash
Copy code
python -m venv venv
Activate the Virtual Environment:

On Windows:
bash
Copy code
venv\Scripts\activate
On macOS/Linux:
bash
Copy code
source venv/bin/activate
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Place the Dataset:

Save the raw Spotify dataset as spotify_dataset.csv in the data/ directory.
Running the Project
Execute the Main Script:

bash
Copy code
python scripts/main.py
Workflow:

Step 1: Preprocesses the raw dataset (preprocess.py).
Step 2: Analyzes the dataset and generates visualizations (analysis.py).
Step 3: Performs clustering to group similar songs (cluster.py).
Step 4: Trains or loads a KMeans model (train.py).
Step 5: Generates personalized recommendations (recommend.py).
Step 6: Creates an HTML report with all outputs and displays it automatically.
Project Structure
bash
Copy code
Spotify/
│
├── data/
│   ├── spotify_dataset.csv          # Raw Spotify dataset
│   ├── preprocessed_data.csv        # Preprocessed dataset
│   ├── clustered_data.csv           # Clustered dataset
│
├── scripts/
│   ├── preprocess.py                # Data preprocessing script
│   ├── analysis.py                  # Exploratory data analysis script
│   ├── cluster.py                   # Clustering and visualization script
│   ├── train.py                     # KMeans training script
│   ├── recommend.py                 # Song recommendation script
│   ├── main.py                      # Main pipeline script
│
├── figures/                         # Visualizations generated during EDA
│
├── clustering_images/               # Visualizations generated during clustering
│
├── report.html                      # Generated HTML report
│
├── requirements.txt                 # List of required Python packages
└── README.md                        # Detailed project documentation
Generated Report
The HTML report includes:

Analysis Visualizations:
Heatmaps, distribution plots, pair plots, etc.
Clustering Visualizations:
Cluster centers, distribution, and pair plots.
Song Recommendations:
A table listing recommended tracks, artists, and genres.
Key Python Scripts
1. preprocess.py
Cleans the raw Spotify dataset.
Encodes categorical columns and scales numerical columns.
Extracts the release_year from the album release date.
2. analysis.py
Generates 10 visualizations to explore the dataset.
Saves visualizations as PNG files in the figures/ directory.
3. cluster.py
Performs KMeans clustering on the dataset.
Visualizes clusters using scatter plots, silhouette scores, and more.
Saves results in the clustering_images/ directory.
4. train.py
Trains or loads a KMeans model.
Saves the trained model as kmeans_model.pkl.
5. recommend.py
Provides song recommendations based on cluster similarity.
Uses the KMeans model to identify clusters for playlists.
Dependencies
Install all dependencies via requirements.txt:

bash
Copy code
pip install -r requirements.txt
pandas (Data manipulation)
matplotlib & seaborn (Data visualization)
scikit-learn (Clustering and machine learning)
jinja2 (HTML template rendering)
joblib (Model persistence)
Troubleshooting
Dataset Errors:

Ensure spotify_dataset.csv is placed in the data/ folder.
Verify that the dataset has the required columns (playlist_id, track_artist, etc.).
Module Import Errors:

Ensure the virtual environment is activated.
Verify the scripts folder is in the same directory as main.py.
Missing Visualizations:

Check the figures/ and clustering_images/ directories for output images.
Ensure sufficient numerical and categorical columns in the dataset.
Future Enhancements
Add advanced models for recommendation, such as collaborative filtering.
Enhance visualizations with interactive dashboards (e.g., Plotly, Dash).
Integrate Spotify API for real-time playlist updates.