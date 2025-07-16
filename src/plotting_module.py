import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_history(model_name, training_history, model_save_dir):
    epochs = training_history['epochs']
    train_losses = training_history['train_losses']
    valid_losses = training_history['valid_losses']

    plt.plot(range(epochs), train_losses, color='blue', label='train loss')
    plt.plot(range(epochs), valid_losses, color='red', label='val loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} training history')

    os.makedirs(model_save_dir, exist_ok=True)
    png_filepath = os.path.join(model_save_dir, f'{model_name}_train_val_losses.png')
    if os.path.exists(png_filepath):
        os.remove(png_filepath)  # Remove the file if it already exists

    plt.savefig(png_filepath, dpi=300)
    print('Saved training history to', png_filepath)
def plot_results(results_df, is_anomalies, anomaly_windows, result_directory, file_name_prefix='', raw_nab_score=None,
                 normalized_nab_score=None, model='ANN'):
    """
    Plots the anomaly detection results, highlighting ground truth anomalies and predicted anomalies.

    Parameters:
    - results_df (DataFrame): DataFrame containing anomaly results and 5XX count data.
    - is_anomalies (Series): Binary series indicating predicted anomalies.
    - anomaly_windows (DataFrame): DataFrame containing ground truth anomaly windows.
    - result_directory (str): Directory to save the plot.
    - file_name_prefix (str): Prefix for the saved file name.
    - raw_nab_score (float): Raw NAB score for the results.
    - normalized_nab_score (float): Normalized NAB score for the results.
    - model (str): Model name used for anomaly detection.
    """
    # Create a figure for the plot
    plt.figure(figsize=(20, 8))

    # Normalize the 5XX count column to a range between 0 and 1 for better visualization
    results_df['5XX_count_normalized'] = (
            (results_df['5XX_count'] - results_df['5XX_count'].min()) /
            (results_df['5XX_count'].max() - results_df['5XX_count'].min())
    )

    # Plot normalized 5XX counts
    plt.plot(results_df.index, results_df['5XX_count_normalized'], label='5XX Count (Normalized)', color='blue',
             alpha=0.6)

    # Set axis labels with appropriate font sizes
    plt.ylabel('5XX Count (Normalized)', fontsize=14)
    plt.xlabel('Time', fontsize=14)

    # # Convert anomaly window columns to datetime format
    # Create a copy to avoid SettingWithCopyWarning
    anomaly_windows = anomaly_windows.copy()

    # Convert anomaly window columns to datetime format
    anomaly_windows.loc[:, 'anomaly_window_start'] = pd.to_datetime(anomaly_windows['anomaly_window_start'])
    anomaly_windows.loc[:, 'anomaly_window_end'] = pd.to_datetime(anomaly_windows['anomaly_window_end'])

    # Map source labels to anomaly labels
    # Map numeric anomaly sources to human-readable labels
    source_map = {1: 'IssueTracker', 2: 'InstantMessenger', 3: 'TestLog'}
    anomaly_windows.loc[:, 'anomaly_label'] = anomaly_windows['anomaly_source'].map(source_map)

    # Plot ground truth anomalies using shaded areas with different colors for each source
    anomaly_sources = ['IssueTracker', 'InstantMessenger', 'TestLog']
    colors = ['green', 'orange', 'purple']
    legend_labels_used = set()

    for source, color in zip(anomaly_sources, colors):
        source_anomalies = anomaly_windows[anomaly_windows['anomaly_label'] == source]
        for _, row in source_anomalies.iterrows():
            label = f'{source} Anomaly' if source not in legend_labels_used else ''
            plt.axvspan(row['anomaly_window_start'], row['anomaly_window_end'], color=color, alpha=0.1, label=label)
            legend_labels_used.add(source)

    # Mark predicted anomalies with red 'X' markers
    plt.scatter(
        results_df.index[is_anomalies == 1],
        results_df['5XX_count_normalized'][is_anomalies == 1],
        color='red', marker='x', label='Predicted Anomalies'
    )

    # Set the plot title and include NAB scores if available
    title = f'Anomaly Detection Results - {model}'
    if raw_nab_score is not None and normalized_nab_score is not None:
        title += f'\nRaw NAB Score: {raw_nab_score:.2f}, Normalized NAB Score: {normalized_nab_score:.2f}'
    plt.title(title, fontsize=16)

    # Add a legend with a customized font size
    plt.legend(loc='upper right', fontsize=12)

    # Customize tick font sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add gridlines to the plot
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file in the specified directory
    png_filepath = os.path.join(result_directory, f'{file_name_prefix}_{model}_anomaly_detection_results.png')
    if os.path.exists(png_filepath):
        os.remove(png_filepath)  # Remove the file if it already exists
    plt.savefig(png_filepath, dpi=300)  # Save with high DPI for better image quality

    # Show the plot
    plt.show()
