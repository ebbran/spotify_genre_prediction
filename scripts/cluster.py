import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os

def perform_clustering(df, output_dir="clustering_images"):
    """
    Perform clustering on the dataset, generate visualizations, and save them as images.

    Parameters:
        df (DataFrame): The preprocessed dataset.
        output_dir (str): Directory where cluster visualizations will be saved.

    Returns:
        df (DataFrame): The dataset with the 'cluster' column added.
        kmeans (KMeans): The trained KMeans model.
        figure_paths (list): Paths to the saved figures.
        figures (list): List of matplotlib figure objects.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List to store paths of saved figures and figure objects
    figure_paths = []
    figures = []

    # Features for clustering
    numerical_columns = ['danceability', 'energy', 'loudness', 'track_popularity']
    X = df[numerical_columns]

    # Perform KMeans clustering
    n_clusters = 5  # Specify number of clusters
    print(f"Performing KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Print cluster centers
    print("Cluster Centers:")
    print(pd.DataFrame(kmeans.cluster_centers_, columns=numerical_columns))

    # **1. Scatter Plot: Clusters and Centers**
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='danceability', y='energy', hue='cluster', palette='tab10')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=150, label='Cluster Centers')
    plt.title('Clusters and Cluster Centers')
    plt.legend()
    cluster_center_path = os.path.join(output_dir, "cluster_centers.png")
    plt.savefig(cluster_center_path)
    figure_paths.append(cluster_center_path)
    figures.append(plt.gcf())
    plt.close()

    # **2. Silhouette Score**
    silhouette_avg = silhouette_score(X, df['cluster'])
    calinski_harabasz = calinski_harabasz_score(X, df['cluster'])
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz}")

    # Silhouette Score Bar Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Silhouette Score'], y=[silhouette_avg], palette='coolwarm')
    plt.ylim(0, 1)  # Silhouette score ranges between -1 and 1
    plt.title('Silhouette Score for Clustering')
    silhouette_path = os.path.join(output_dir, "silhouette_score.png")
    plt.savefig(silhouette_path)
    figure_paths.append(silhouette_path)
    figures.append(plt.gcf())
    plt.close()

    # **3. Cluster Distribution**
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cluster', palette='Set2')
    plt.title('Cluster Distribution')
    cluster_dist_path = os.path.join(output_dir, "cluster_distribution.png")
    plt.savefig(cluster_dist_path)
    figure_paths.append(cluster_dist_path)
    figures.append(plt.gcf())
    plt.close()

    # **4. Pair Plot: Features by Cluster**
    pair_plot_path = os.path.join(output_dir, "pair_plot_clusters.png")
    pair_plot = sns.pairplot(df, vars=numerical_columns[:4], hue='cluster', palette='tab10')
    pair_plot.savefig(pair_plot_path)
    figure_paths.append(pair_plot_path)
    figures.append(pair_plot.fig)
    plt.close()

    print(f"Figures created: {len(figures)}")

    # Return results
    return df, kmeans, figure_paths, figures


# Usage in main.py
def main():
    # Load the preprocessed dataset
    data_path = 'data/preprocessed_data.csv'
    output_dir = 'clustering_images'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Perform clustering and save figures
    df, kmeans, figure_paths, figures = perform_clustering(df, output_dir=output_dir)

    # Log final clustering summary
    print("Clustering completed successfully.")
    print(f"Number of Clusters: {kmeans.n_clusters}")
    print(f"Silhouette Score: {silhouette_score(df[['danceability', 'energy', 'loudness', 'track_popularity']], df['cluster'])}")

if __name__ == "__main__":
    main()
