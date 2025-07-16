import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# from preprocessing import create_windowed_data  # Import Create Windowed Data function


def compute_anomaly_likelihood(historical_recon_error, anomaly_long_window_length, anomaly_short_window_length):
    """
    Computes the anomaly likelihood based on reconstruction error and statistical analysis.

    Parameters:
    - historical_recon_error: Array, historical reconstruction error values.
    - anomaly_long_window_length: Integer, length of the long window for analysis.
    - anomaly_short_window_length: Integer, length of the short window for analysis.

    Returns:
    - likelihood: Float, computed anomaly likelihood.
    """
    if len(historical_recon_error) <= anomaly_long_window_length:
        return 0.5

    wide_data_window = historical_recon_error[-anomaly_long_window_length:]
    narrow_data_window = historical_recon_error[-anomaly_short_window_length:]

    mean_of_wide_data_window = np.mean(wide_data_window)
    stdev_of_wide_data_window = np.std(wide_data_window)

    # Fixed epsilon to avoid division by zero
    epsilon = 1e-10
    likelihood = 0.5 + 0.5 * (
        1 - norm.sf(
            (np.mean(narrow_data_window) - mean_of_wide_data_window) / (stdev_of_wide_data_window + epsilon)
        )
    )

    return likelihood


# def detect_anomalies_with_likelihood(autoencoder, X_test, anomaly_threshold, long_window, short_window, index, model_type='AE', time_steps=1, labels=None):
#     """
#     Detects anomalies in the test data using the trained autoencoder and computed likelihood.
#
#     Parameters:
#     - autoencoder: Keras Model, trained autoencoder model.
#     - X_test: Array, test data.
#     - anomaly_threshold: Float, threshold for detecting anomalies.
#     - long_window: Integer, length of the long window for likelihood computation.
#     - short_window: Integer, length of the short window for likelihood computation.
#     - index: Index, index of the test data.
#     - model_type: String, type of model ('AE' or 'GRU').
#     - time_steps: Integer, length of time steps for sequential models.
#
#     Returns:
#     - is_anomalies: Series, binary series indicating whether each point is an anomaly.
#     - likelihoods: Array, computed anomaly likelihoods.
#     - reconstruction_error: Array, computed reconstruction errors.
#     """
#
#
#     # Reshape X_test if using GRU or LSTM model
#     if model_type == 'GRU' or model_type == 'LSTM':
#         input_dim = X_test.shape[1]
#         # Ensure time_steps matches the model's expected input
#         X_test_windowed, missing_points = create_windowed_data(X_test, time_steps)
#         X_test_windowed = X_test_windowed.reshape(-1, time_steps, input_dim)
#
#         # Make predictions
#         X_test_predictions = autoencoder.predict(X_test_windowed)
#
#         # Reshape predictions back
#         X_test_predictions = X_test_predictions.reshape(-1, input_dim)
#
#         # Compute reconstruction error
#         # window_reconstruction_error = np.mean(
#         #     np.power(X_test_windowed.reshape(-1, input_dim) - X_test_predictions, 2), axis=1
#         # )
#         reconstruction_error_raw = np.abs(X_test_windowed.reshape(-1, input_dim) - X_test_predictions)
#         window_reconstruction_error = np.max(reconstruction_error_raw**2, axis=1)
#
#         # Initialize full reconstruction error with zeros
#         reconstruction_error_full = np.zeros(len(X_test))
#
#         # Average the windowed errors over the overlapping regions
#         counts = np.zeros(len(X_test))
#         for i in range(len(window_reconstruction_error)):
#             reconstruction_error_full[i:i + time_steps] += window_reconstruction_error[i]
#             counts[i:i + time_steps] += 1
#
#         # Avoid division by zero
#         counts[counts == 0] = 1
#         reconstruction_error_full /= counts
#
#     elif model_type == 'GDN':
#         X_test.index = index
#         autoencoder.fit(None, X_test, labels, mode="only_predict")
#         # autoencoder._set_seeds()
#         # autoencoder._load_only_test_data(X_test)
#         # autoencoder._load_model()
#         # autoencoder._get_model_path()
#         # # self._train()
#         # autoencoder._get_score()
#
#         X_test_predictions = autoencoder.X_test_predictions
#         print(f'X_test_predictions.shape: {X_test_predictions.shape}')
#         # reconstruction_error_full = np.mean(
#         #     np.power(X_test - X_test_predictions, 2), axis=1
#         # )
#         reconstruction_error_raw = np.abs(X_test_predictions - X_test)
#         reconstruction_error_full = np.max(reconstruction_error_raw ** 2, axis=1)
#         # reconstruction_error_raw = MinMaxScaler().fit_transform(reconstruction_error_raw)
#         # reconstruction_error_full = np.sort(reconstruction_error_raw, axis=1)[:, -3:].mean(axis=1)
#
#         # reconstruction_error = autoencoder.test_err_scores.T
#         # reconstruction_error_full = np.mean(
#         #     np.power(reconstruction_error, 2), axis=1
#         # )
#         # reconstruction_error_full = autoencoder.topk_err_scores
#         # print(f'reconstruction_error_full.shape', reconstruction_error_full.shape)
#     elif model_type == 'A3TGCN':
#         # For ANN models
#         X_test_predictions, reconstruction_error_raw, test_loss = autoencoder.predict(X_test)
#         # reconstruction_error_full = np.mean(
#         #     np.power(X_test - X_test_predictions, 2), axis=1
#         # )
#         # reconstruction_error_raw = np.abs(X_test_predictions - X_test)
#         # reconstruction_error_normalized = MinMaxScaler().fit_transform(reconstruction_error_raw)
#         reconstruction_error_raw = reconstruction_error_raw.max(axis=2)
#         reconstruction_error_full = np.max(reconstruction_error_raw ** 2, axis=1)
#     else:
#         # For ANN models
#         X_test_predictions = autoencoder.predict(X_test)
#         # reconstruction_error_full = np.mean(
#         #     np.power(X_test - X_test_predictions, 2), axis=1
#         # )
#         reconstruction_error_raw = np.abs(X_test_predictions - X_test)
#         # reconstruction_error_normalized = MinMaxScaler().fit_transform(reconstruction_error_raw)
#         reconstruction_error_full = np.max(reconstruction_error_raw ** 2, axis=1)
#
#
#     # Compute anomaly likelihoods
#     likelihoods = []
#     for i in range(len(reconstruction_error_full)):
#         likelihood = compute_anomaly_likelihood(
#             reconstruction_error_full[:i + 1], long_window, short_window
#         )
#         likelihoods.append(likelihood)
#
#     likelihoods = np.array(likelihoods)
#     is_anomalies = (likelihoods > anomaly_threshold).astype(int)
#
#     return pd.Series(is_anomalies, index=index), likelihoods, reconstruction_error_full

# def detect_anomalies_with_likelihood_shorten(autoencoder, X_test, anomaly_threshold, long_window, short_window, index, model_type='AE', time_steps=1):
#     """
#     Detects anomalies in the test data using the trained autoencoder and computed likelihood.
#
#     Parameters:
#     - autoencoder: Keras Model, trained autoencoder model.
#     - X_test: Array, test data.
#     - anomaly_threshold: Float, threshold for detecting anomalies.
#     - long_window: Integer, length of the long window for likelihood computation.
#     - short_window: Integer, length of the short window for likelihood computation.
#     - index: Index, index of the test data.
#     - model_type: String, type of model ('AE' or 'GRU').
#     - time_steps: Integer, length of time steps for sequential models.
#
#     Returns:
#     - is_anomalies: Series, binary series indicating whether each point is an anomaly.
#     - likelihoods: Array, computed anomaly likelihoods.
#     - reconstruction_error: Array, computed reconstruction errors.
#     """
#     input_dim = X_test.shape[1]
#
#     # Reshape X_test if using GRU or LSTM model
#     if model_type == 'GRU' or model_type == 'LSTM':
#         # Ensure time_steps matches the model's expected input
#         X_test_windowed, missing_points = create_windowed_data(X_test, time_steps)
#         X_test_windowed = X_test_windowed.reshape(-1, time_steps, input_dim)
#
#         # Make predictions
#         X_test_predictions = autoencoder.predict(X_test_windowed)
#
#         # Reshape predictions back
#         X_test_predictions = X_test_predictions.reshape(-1, input_dim)
#
#         # Compute reconstruction error
#         window_reconstruction_error = np.mean(
#             np.power(X_test_windowed.reshape(-1, input_dim) - X_test_predictions, 2), axis=1
#         )
#
#         # Initialize full reconstruction error with zeros
#         reconstruction_error_full = np.zeros(len(X_test))
#
#         # Average the windowed errors over the overlapping regions
#         counts = np.zeros(len(X_test))
#         for i in range(len(window_reconstruction_error)):
#             reconstruction_error_full[i:i + time_steps] += window_reconstruction_error[i]
#             counts[i:i + time_steps] += 1
#
#         # Avoid division by zero
#         counts[counts == 0] = 1
#         reconstruction_error_full /= counts
#     else:
#         # For ANN models
#         # X_test_predictions = autoencoder.predict(X_test)
#         # X_test = pd.read_csv('/home/thudth/PycharmProjects/gnnad/tests/X_test.csv').values
#         X_test_predictions = pd.read_parquet('/home/thudth/PycharmProjects/gnnad/tests/test_err_scores.parquet').values.T
#         print(f'X_test.shape', X_test.shape, 'X_test_predictions.shape', X_test_predictions.shape)
#         reconstruction_error_full = np.mean(
#             np.power(X_test - X_test_predictions, 2), axis=1
#         )
#
#         reconstruction_error_full = pd.read_csv('/home/thudth/PycharmProjects/gnnad/tests/topk_err_scores.csv').values[:,0]
#
#     # Compute anomaly likelihoods
#     likelihoods = []
#     for i in range(len(reconstruction_error_full)):
#         likelihood = compute_anomaly_likelihood(
#             reconstruction_error_full[:i + 1], long_window, short_window
#         )
#         likelihoods.append(likelihood)
#
#     likelihoods = np.array(likelihoods)
#     is_anomalies = (likelihoods > anomaly_threshold).astype(int)
#
#     return pd.Series(is_anomalies, index=index), likelihoods, reconstruction_error_full
