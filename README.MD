# ClouDens: A Scalable Ensemble Framework for Anomaly Detection in Large-Scale Cloud Systems

[![dataset DOI](https://img.shields.io/badge/dataset%20DOI-10.5281%2Fzenodo.14062900-blue.svg)](https://doi.org/10.5281/zenodo.14062900)
![python](https://img.shields.io/badge/python-3.12.2-blue.svg)


This repository contains the code implementation for the paper submitted in ICSE Research Track 2026 with title **"ClouDens: A Scalable Ensemble Framework for Anomaly Detection in Large-Scale Cloud Systems"**.

The repository includes scripts and modules for anomaly detection using Spatio-Temporal Graph Neural Networks (ST-GNNs), NAB scoring, and related preprocessing tasks. 
The pipeline is modularized for flexibility and ease of use.

# Project Setup: Anomaly Data Preparation

This guide helps you set up the environment for the "Anomaly Detection" project. Follow these steps to ensure you have the correct Python version and dependencies installed for replicability.

---

## 1 Requirements

- **Python Version**: Python 3.12.2
- **Operating System**: Windows, macOS, or Linux
- **Tools**:
  - Python installed on your system
  - `pip` for package management
  - A terminal or command-line interface

---

## 2 Setup Instructions



### 2.1 Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/ClouDens/icse2026_cloudens_ad.git
```

Navigate to the project directory:
```bash
cd icse2026_cloudens_ad
```

### 2.2 Verify Python Version
Ensure you have Python 3.12.2 installed:
```bash
python --version
```
If Python 3.12.2 is not installed, download it from the [official Python website](https://www.python.org/downloads/release/python-3122/) and install it.


### Setup Options

> **Using Docker**:  
> All dependencies, directory structures, and data downloads are handled automatically. Refer to [Section 5.1 Using Docker](#51-using-docker) for execution details.

> **Using a Virtual Environment**:  
> Continue with the steps below and refer to [Section 5.2 Using Virtual Environment](#52-using-virtual-environment) for execution instructions.


### 2.3 Create a Virtual Environment
Create a virtual environment using Python 3.12.2:
```bash
python -m venv venv
```

### 2.4 Activate the Virtual Environment
Activate the virtual environment:

- **On Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **On macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

Verify that the virtual environment is using Python 3.12.2:
```bash
python --version
```

### 2.5 Install Dependencies
Install all required libraries from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2.6 Run Tests
Run a test script or a few commands from the project to ensure everything is working correctly.

### 2.7 Optional: Update Dependencies
If additional libraries are needed, install them and update `requirements.txt`:
```bash
pip install <library-name>
pip freeze > requirements.txt
```

### 2.8 Directory Structure

Use the following directory structure for your project:

```plaintext
icse2026_cloudens_ad/
├── conf/                # Configuration files (e.g., config.yaml)
├── src/                 # Source code files
├── data/
│   ├── massaged/        # Pivoted input data
│   ├── labels/          # Anomaly window labels
├── results/
│   ├── model_experiments/ # Experiment results
├── trained_models/      # Saved trained models
```

You can **create these directories** using the following shell script:

```bash
mkdir -p conf src data/massaged data/labels results/model_experiments trained_models
```


## 3 Data Source
The data required for this project is provided in the following dataset:
> Islam, M. S., Rakha, M. S., Pourmajidi, W., Sivaloganathan, J., Steinbacher, J., & Miranskyy, A. (2024).
> Dataset for the paper "Anomaly Detection in Large-Scale Cloud Systems: An Industry Case and Dataset" (v1.0) [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.14062900

### 3.1 Input Data

- Place pivoted data files (e.g., `pivoted_data_all.parquet`) in `data/massaged/`.
- Place anomaly window labels `anomaly_windows.csv` in `data/labels/`.

You can achieve this using the following commands in the terminal (on macOS/Linux):
```shell
curl -L -o data/labels/anomaly_windows.csv https://zenodo.org/records/14062900/files/anomaly_windows.csv?download=1
curl -L -o data/massaged/pivoted_data_all.parquet https://zenodo.org/records/14062900/files/pivoted_data_all.parquet?download=1
```


---
> **For manual setup using a virtual environment**, after downloading the files, proceed to [Section 5.2 Using Virtual Environment](#52-using-virtual-environment) for execution details.


## 4 File Descriptions

### 4.1 Configuration

- **`config.yaml`**: Stores the configuration for file paths, training/testing parameters, and model settings.

### 4.2 Scripts

- **`run_experiment__multi_models_GRU_ANN.py`**: The main script for coordinating anomaly detection experiments. It handles data preprocessing, model training, evaluation, and grid search for optimizing parameters.

- **`preprocessing.py`**: Handles data preparation, including loading 5XX features, adding time-related features (e.g., sine/cosine transformations), and filtering training/testing anomaly windows.

- **`anomaly_likelihood.py`**: Contains functions to compute anomaly likelihood using reconstruction errors and statistical analysis.

- **`nab_scoring.py`**: Implements NAB (Numenta Anomaly Benchmark) scoring, including options for standard scoring or custom profiles like `reward_fn`.

- **`plotting_module.py`**: Provides utilities for visualizing the results of anomaly detection, such as normalized 5XX counts and detected anomalies.


### 4.3 Results Directory (`./results/model_experiments/`)

#### Files:

- **`unweighted__<Model>_anomaly_detection_results.csv`**  
   Contains detailed results from the unweighted anomaly detection experiments using the specified model (e.g., ANN or GRU), including metrics such as true positives, false positives, and anomaly windows, for the entire test period.

- **`unweighted__<Model>_anomaly_detection_results.png`**  
   A graphical visualization of the results from the unweighted anomaly detection experiment, depicting anomalies and their corresponding NAB scores. The plot includes:
   - **5XX Count (Normalized)**: The normalized count of 5XX errors.
   - **Predicted Anomalies (Red X)**: Anomalies detected by the model.
   - **Ground Truth Anomalies**: Highlighted areas based on categories like IssueTracker, InstantMessenger, and TestLog.

- **`unweighted__<Model>_results.csv`**  
   Consolidated results of the unweighted experiments with the specified model, providing a summary of key performance metrics.

---

## 5 Execution

### 5.1 Using Docker

#### 5.1.1 Build the Docker image using the provided `Dockerfile`:
```bash
docker build -t anomaly-detector .
```

#### 5.1.2 Run the Docker container:
```bash
docker run -it anomaly-detector
```

### 5.2 Using Virtual Environment

The workflow is configured using the [Hydra](https://github.com/facebookresearch/hydra) framework.

#### 5.2.1 Configuring the Model Type

The model type (e.g., ANN or GRU) is configurable in the `conf.yaml` file. Modify the following parameters under `train_test_config`:

```yaml
train_test_config:
  use_model: ANN                 # Options: ANN or GRU
```

#### 5.2.2 Configuring NAB Scoring Profile

The NAB scoring profile used for evaluation can be configured in the `conf.yaml` file. Update the following parameters under the `evaluation` section:

```yaml
evaluation:  # Default parameters
  nab_scoring_profile: "reward_fn"  # Options: "standard" or "reward_fn"
```


#### 5.2.3 Run the Project
To run the project, execute the main script after setting up the required directories and input files:

```bash
python src/run_experiment__multi_models_GRU_ANN.py
```

Note that you can pass configuration parameters directly via the command line. For example:
```bash
python src/run_experiment__multi_models_GRU_ANN.py train_test_config.use_model=GRU evaluation.nab_scoring_profile=reward_fn  
```

---

## Notes for Replicability

- Use Python 3.12.2 to avoid compatibility issues.
- Keep the `requirements.txt` file up-to-date if new dependencies are added.
- Use the exact steps mentioned above to ensure a consistent environment across different setups.

---

If you encounter any issues, please feel free to reach out for support by opening an issue.

