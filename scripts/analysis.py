import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(df, output_folder="figures"):
    """
    Perform detailed analysis on the dataset and generate 10 visualizations.
    Saves each figure as an image file in the specified output folder and 
    returns the figures along with the image file paths.

    Parameters:
        df (DataFrame): The preprocessed dataset.
        output_folder (str): Folder where image files will be saved.

    Returns:
        image_paths (list): A list of file paths to the saved images.
        figures (list): A list of matplotlib figure objects.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List to hold paths of saved images and figure objects
    image_paths = []
    figures = []

    # Filter numerical columns for analysis
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # **1. Correlation Heatmap**
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    image_paths.append(heatmap_path)
    figures.append(plt.gcf())
    plt.close()

    # **2. Distribution Plot (Histogram)**
    plt.figure(figsize=(12, 6))
    for col in numerical_columns[:5]:  # Limit to the first 5 numerical columns
        sns.histplot(df[col], kde=True, label=col, bins=30, alpha=0.6)
    plt.legend()
    plt.title('Distribution of Numerical Features')
    dist_path = os.path.join(output_folder, "feature_distribution.png")
    plt.savefig(dist_path)
    image_paths.append(dist_path)
    figures.append(plt.gcf())
    plt.close()

    # **3. Pair Plot**
    pair_plot_path = os.path.join(output_folder, "pair_plot.png")
    pair_plot = sns.pairplot(df[numerical_columns[:4]])  # Limit to 4 features for clarity
    pair_plot.savefig(pair_plot_path)
    image_paths.append(pair_plot_path)
    figures.append(pair_plot.fig)
    plt.close()

    # **4. Box Plot (Outliers Analysis)**
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[numerical_columns], palette="Set3")
    plt.title('Boxplot of Numerical Features')
    boxplot_path = os.path.join(output_folder, "boxplot_features.png")
    plt.savefig(boxplot_path)
    image_paths.append(boxplot_path)
    figures.append(plt.gcf())
    plt.close()

    # **5. Count Plot (Categorical Analysis)**
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=categorical_columns[0], palette="Set2", order=df[categorical_columns[0]].value_counts().index)
        plt.title(f'Count Plot for {categorical_columns[0]}')
        countplot_path = os.path.join(output_folder, "count_plot.png")
        plt.savefig(countplot_path)
        image_paths.append(countplot_path)
        figures.append(plt.gcf())
        plt.close()

    # **6. Violin Plot**
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x="track_popularity", y="energy", palette="muted")
    plt.title('Violin Plot: Track Popularity vs Energy')
    violin_path = os.path.join(output_folder, "violin_plot.png")
    plt.savefig(violin_path)
    image_paths.append(violin_path)
    figures.append(plt.gcf())
    plt.close()

    # **7. Scatter Plot**
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="danceability", y="energy", hue="track_popularity", palette="coolwarm", size="track_popularity", sizes=(20, 200))
    plt.title('Scatter Plot: Danceability vs Energy (Colored by Popularity)')
    scatter_path = os.path.join(output_folder, "scatter_plot.png")
    plt.savefig(scatter_path)
    image_paths.append(scatter_path)
    figures.append(plt.gcf())
    plt.close()

    # **8. KDE Plot**
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="danceability", y="energy", cmap="Blues", fill=True)
    plt.title('KDE Plot: Danceability vs Energy')
    kde_path = os.path.join(output_folder, "kde_plot.png")
    plt.savefig(kde_path)
    image_paths.append(kde_path)
    figures.append(plt.gcf())
    plt.close()

    # **9. Line Plot**
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df.sort_values("track_popularity"), x="track_popularity", y="energy", label="Energy")
    sns.lineplot(data=df.sort_values("track_popularity"), x="track_popularity", y="danceability", label="Danceability")
    plt.legend()
    plt.title('Line Plot: Popularity vs Energy/Danceability')
    lineplot_path = os.path.join(output_folder, "line_plot.png")
    plt.savefig(lineplot_path)
    image_paths.append(lineplot_path)
    figures.append(plt.gcf())
    plt.close()

    # **10. Bar Plot**
    plt.figure(figsize=(12, 6))
    top_artists = df['track_artist'].value_counts().head(10)
    sns.barplot(x=top_artists.values, y=top_artists.index, palette="coolwarm")
    plt.title('Top 10 Artists by Track Count')
    barplot_path = os.path.join(output_folder, "bar_plot.png")
    plt.savefig(barplot_path)
    image_paths.append(barplot_path)
    figures.append(plt.gcf())
    plt.close()

    print(f"Figures created: {len(figures)}")
    return image_paths, figures
