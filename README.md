# DynamicLSTM AutoML-LSTM Time Series Dataset Analysis and Model Training App

## This is a Streamlit application for analyzing time series datasets and training custom LSTM models for regression tasks.

## Features

- **Dataset Analysis**: Analyze time series datasets by extracting various properties such as the number of records, columns, textual, numeric, and date columns, as well as highly dependent columns.
- **Correlation Matrix**: Visualize the correlation matrix and heatmap of numeric columns in the dataset.
- **Missing Values Handling**: Handle missing values in numeric columns using methods such as mean, median, mode, or dropping rows with missing values.
- **Column Dropping**: Allow users to drop selected columns from the dataset.
- **Model Training**: Train LSTM regression models using dynamically determined hyperparameters based on dataset properties, including LSTM layers, dense layers, return sequence, bidirectional, and dropout.
- **Loss Plotting**: Plot the training and validation loss versus epoch graph to visualize model training progress.
- **Model Download**: Provide a download button to download the trained model.

## How It Helps Developers

### Automated Data Analysis

Developers can quickly analyze time series datasets without the need for manual coding. The application automatically extracts key dataset properties and provides insights into the data distribution, saving developers valuable time and effort.

### Efficient Data Preprocessing

Built-in functionality for handling missing values and dropping columns allows developers to efficiently preprocess datasets for model training. This eliminates the need for writing custom code for data cleaning and ensures a streamlined preprocessing pipeline.

### Hyperparameter Tuning Automation

The application dynamically determines hyperparameters for LSTM regression models based on dataset properties. This automation reduces the burden of manual hyperparameter tuning, allowing developers to focus on model architecture and experimentation.

### Visual Model Training Monitoring

Developers can visualize the training and validation loss versus epoch graph during model training. This graphical representation helps developers monitor the model's performance and convergence, enabling them to make informed decisions and adjustments as needed.

### Easy Model Deployment

Upon completion of training, developers can download the trained LSTM regression model for deployment in production environments. This facilitates seamless integration of the model into applications or systems without the need for additional training.

### Streamlined Development Workflow

The user-friendly interface of the Streamlit application offers a seamless development experience. Developers can navigate through various functionalities effortlessly, reducing development time and enhancing productivity.


## Usage

1. **Upload Dataset**: Upload a CSV file containing time series data.
2. **Dataset Analysis**: View dataset properties, correlation matrix, and missing values counts.
3. **Data Preprocessing**: Handle missing values and drop columns as needed.
4. **Model Training**: Select features for input and output, choose scaler options, and train the LSTM regression model.
5. **Model Evaluation**: View training and validation loss, and download the trained model for future use.

## Direct Access
You can directly utilize this application via this [link](https://lstmdynamicsidd.streamlit.app/).

## Contributions
Contributions are welcome! Please feel free to open issues or pull requests for any improvements or bug fixes.
