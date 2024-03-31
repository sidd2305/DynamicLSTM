import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from dateutil.parser import parse
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense

global epoch,batch,drop,returseq,bidi
epoch=0
batch=0
drop=0
returnseq=0
bidi=0


url = "https://raw.githubusercontent.com/sidd2305/DynamicLSTM/5e054d621262f5971ba1a5b54d8e7ec6b9573baf/hu.csv"
dataset = pd.read_csv(url)

class KNNUnsupervised:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = tf.constant(X, dtype=tf.float32)
        self.y_train = tf.constant(y, dtype=tf.float32)

    def predict(self, X):
        X_test = tf.constant(X, dtype=tf.float32)
        distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test, axis=1) - self.X_train), axis=2)
        top_k_indices = tf.argsort(distances, axis=1)[:, :self.k]
        nearest_neighbor_labels = tf.gather(self.y_train, top_k_indices, axis=0)

        # Calculate average values of specified columns for nearest neighbors
        avg_values = tf.reduce_mean(nearest_neighbor_labels, axis=1)

        return avg_values.numpy()

class KNNUnsupervisedLSTM:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        # Convert string representation of LSTM units to numeric arrays
        max_layers = 0
        y_processed = []
        for units in y[:, 5]:  # Assuming LSTM units are in the 5th column
            units_array = eval(units) if isinstance(units, str) else [units]
            max_layers = max(max_layers, len(units_array))
            y_processed.append(units_array)
        
        # Pad arrays with zeros to ensure uniform length
        for i in range(len(y_processed)):
            y_processed[i] += [0] * (max_layers - len(y_processed[i]))

        # Convert input and output arrays to TensorFlow constant tensors
        self.X_train = tf.constant(X, dtype=tf.float32)
        self.y_train = tf.constant(y_processed, dtype=tf.float32)

    def predict(self, X):
        X_test = tf.constant(X, dtype=tf.float32)
        distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test, axis=1) - self.X_train), axis=2)
        top_k_indices = tf.argsort(distances, axis=1)[:, :self.k]
        nearest_neighbor_labels = tf.gather(self.y_train, top_k_indices, axis=0)
        neighbor_indices = top_k_indices.numpy()

        # Calculate average values of specified columns for nearest neighbors
        avg_values = tf.reduce_mean(nearest_neighbor_labels, axis=1)
        
        return avg_values.numpy(), neighbor_indices


def split_data(sequence, n_steps):
    X, Y = [], []
    for i in range(len(sequence) - n_steps):
        x_seq = sequence[i:i + n_steps]
        y_seq = sequence.iloc[i + n_steps]
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)

def handle_date_columns(dat, col):
    # Convert the column to datetime
    dat[col] = pd.to_datetime(dat[col], errors='coerce')
    # Extract date components
    dat[f'{col}_year'] = dat[col].dt.year
    dat[f'{col}_month'] = dat[col].dt.month
    dat[f'{col}_day'] = dat[col].dt.day
    # Extract time components
    dat[f'{col}_hour'] = dat[col].dt.hour
    dat[f'{col}_minute'] = dat[col].dt.minute
    dat[f'{col}_second'] = dat[col].dt.second
def is_date(string):
    try:
        # Check if the string can be parsed as a date
        parse(string)
        return True
    except ValueError:
        # If parsing fails, also check if the string matches a specific date format
        return bool(re.match(r'^\d{2}-\d{2}-\d{2}$', string))

def analyze_csv(df):
    # Get the number of records
    num_records = len(df)

    # Get the number of columns
    num_columns = len(df.columns)

    # Initialize counters for textual, numeric, and date columns
    num_textual_columns = 0
    num_numeric_columns = 0
    num_date_columns = 0

    # Identify textual, numeric, and date columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            if all(df[col].apply(is_date)):
                handle_date_columns(df, col)
                num_date_columns += 1
            else:
                num_textual_columns += 1
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_numeric_columns += 1

    # Find highly dependent columns (you may need to define what "highly dependent" means)
    # For example, you can use correlation coefficients or other statistical measures

    # In this example, let's assume highly dependent columns are those with correlation coefficient above 0.8
    highly_dependent_columns = set()
    correlation_matrix = df.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                highly_dependent_columns.add(col1)
                highly_dependent_columns.add(col2)

    num_highly_dependent_columns = len(highly_dependent_columns)

    # Output the results
    st.write("Number Of Records:", num_records)
    st.write("Number Of Columns:", num_columns)
    st.write("Number of Date Columns:", num_date_columns)

    st.write("Number of Textual Columns:", num_textual_columns)
    st.write("Number of Numeric Columns:", num_numeric_columns)

    st.write("Total Number of highly dependent columns:", num_highly_dependent_columns)
    X = dataset[['Number Of Records', 'Number Of Columns',
                 'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
    y = dataset[['Bidirectional', 'Return Sequence=True', 'Dropout', 'Epochs', 'Batch Size']].values

    knn = KNNUnsupervised(k=3)
    knn.fit(X, y)

    # Input data for which we want to predict the average values
    q1 = np.array([[num_records,num_columns,num_textual_columns,num_numeric_columns,num_highly_dependent_columns]])  # Example input data, 1 row, 6 columns
    avg_neighbors = knn.predict(q1)

    # Apply sigmoid to the first two elements
    for i in range(len(avg_neighbors)):
        # avg_neighbors[i][0] = 1 / (1 + np.exp(-avg_neighbors[i][0]))
        # avg_neighbors[i][1] = 1 / (1 + np.exp(-avg_neighbors[i][1]))
        avg_neighbors[i][0] = 1 if avg_neighbors[i][0] >= 0.5 else 0
        avg_neighbors[i][1] = 1 if avg_neighbors[i][1] >= 0.5 else 0

    # st.write("Output using KNN of info 1-Bidirectional,Return Sequence,Dropout,Epochs,BatchSize:")
    # st.write(avg_neighbors)
    # st.write(avg_neighbors.shape)
    global epoch,batch,drop,returseq,bidi
    #poch,batch,drop,returseq,bidi
    epoch=avg_neighbors[0][3]
    batch=avg_neighbors[0][4]
    drop=avg_neighbors[0][2]
    bidi=avg_neighbors[0][0]
    returnseq=avg_neighbors[0][1]
    # st.write("epoch is",epoch)



    #LSTM Layer
    X = dataset[['Number Of Records', 'Number Of Columns', 
                 'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
    y = dataset[['Bidirectional', 'Return Sequence=True', 'Dropout', 'Epochs', 'Batch Size', 'LSTM Layers']].values
    knn1 = KNNUnsupervisedLSTM(k=3)
    knn1.fit(X, y)
    
  
    avg_neighbors, neighbor_indices = knn1.predict(q1)

    # Extract LSTM units of k-nearest neighbors
    lstm_units = y[neighbor_indices[:, 0], 5]  # Extract LSTM units corresponding to the indices of k-nearest neighbors
    lstm_units_array = []
    for units in lstm_units:
        units_list = [int(x) for x in units.strip('[]').split(',')]
        lstm_units_array.append(units_list)

    # Get the maximum length of nested lists
    max_length = max(len(units) for units in lstm_units_array)

    # Pad shorter lists with zeros to match the length of the longest list
    padded_lstm_units_array = [units + [0] * (max_length - len(units)) for units in lstm_units_array]

    # Convert the padded list of lists to a numpy array
    lstm_units_array_transpose = np.array(padded_lstm_units_array).T

    # Calculate the average of each element in the nested lists
    avg_lstm_units = np.mean(lstm_units_array_transpose, axis=1)

    global output_array_l
    output_array_l = np.array(list(avg_lstm_units))
    # st.write("LSTM Layer Output")
    # st.write(output_array_l)

    #Dense Layer thing
    X = dataset[['Number Of Records', 'Number Of Columns', 
                 'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
    y = dataset[['Bidirectional', 'Return Sequence=True', 'Dropout', 'Epochs', 'Batch Size', 'LSTM Layers', 'Dense Layers']].values
    knn = KNNUnsupervisedLSTM(k=3)
    knn.fit(X, y)
    
    
    avg_neighbors, neighbor_indices = knn.predict(q1)

    # Extract Dense layers of k-nearest neighbors
    dense_layers = y[neighbor_indices[:, 0], 6]  # Extract Dense layers corresponding to the indices of k-nearest neighbors
    dense_layers_array = []
    for layers in dense_layers:
        layers_list = [int(x) for x in layers.strip('[]').split(',')]
        dense_layers_array.append(layers_list)

    # Get the maximum length of nested lists
    max_length = max(len(layers) for layers in dense_layers_array)

    # Pad shorter lists with zeros to match the length of the longest list
    padded_dense_layers_array = [layers + [0] * (max_length - len(layers)) for layers in dense_layers_array]

    # Convert the padded list of lists to a numpy array
    dense_layers_array_transpose = np.array(padded_dense_layers_array).T

    # Calculate the average of each element in the nested lists
    avg_dense_layers = np.mean(dense_layers_array_transpose, axis=1)

    global output_array_d
    # Print the output in the form of an array
    output_array_d = np.array(list(avg_dense_layers))
    # st.write("Dense layer output:")
    # st.write(output_array_d)


def load_data(file):
    df = pd.read_csv(file)
    st.subheader("1. Show first 10 records of the dataset")
    st.dataframe(df.head(10))
    analyze_csv(df)
     # Call analyze_csv function here

    return df

def show_correlation(df):
    st.subheader("2. Show the correlation matrix and heatmap")
    numeric_columns = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numeric_columns].corr()
    st.dataframe(correlation_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    st.pyplot(fig)

def show_missing_values(df):
    st.subheader("3. Show the number of missing values in each column")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values)
    st.write(output_array_d)

def handle_missing_values(df):
    st.subheader("4. Handle missing values")
    numeric_columns = df.select_dtypes(include=['number']).columns

    fill_option = st.radio("Choose a method to handle missing values:", ('Mean', 'Median', 'Mode', 'Drop'))

    if fill_option == 'Drop':
        df = df.dropna(subset=numeric_columns)
    else:
        fill_value = (
            df[numeric_columns].mean() if fill_option == 'Mean'
            else (df[numeric_columns].median() if fill_option == 'Median'
                  else df[numeric_columns].mode().iloc[0])
        )
        df[numeric_columns] = df[numeric_columns].fillna(fill_value)

    st.dataframe(df)

    return df

def drop_column(df):
    st.subheader("5. Drop a column")
    columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        st.dataframe(df)

    return df

def build_model(layer_sizes, dense_layers, return_sequence, bidirectional, dropout):
    model = tf.keras.Sequential()
    
    for i, size in enumerate(layer_sizes):
        size = int(size) 
        if i == 0:
            # For the first layer, we need to specify input_shape
            # model.add(LSTM(size, return_sequences=bool(return_sequence))) then did  model.add(LSTM(size,input_shape=(c,d), return_sequences=True))
            model.add(LSTM(size,input_shape=(X_train.shape[1], 1), return_sequences=True))
            
        else:
            if bool(bidirectional):  # Bidirectional layer
                model.add(Bidirectional(LSTM(size, return_sequences=True)))
            else:
                model.add(LSTM(size,return_sequences=True))

        if dropout > 0:  # Dropout
            model.add(Dropout(dropout))

    for nodes in dense_layers:
        model.add(Dense(nodes, activation='relu'))

    model.add(Dense(1))  # Example output layer, adjust as needed

    model.compile(optimizer='adam', loss='mse')  # Compile the model
    model.build((None, None, layer_sizes[0]))  # Explicitly build the model

    return model

def train_regression_model(df):
    st.subheader("6. Train a Custom Time Series model")

    if df.empty:
        st.warning("Please upload a valid dataset.")
        return

    st.write("Select columns for X (features):")
    st.write("Please DO NOT select your date column.We have automatically pre processed it into date,month,year(hour,min,sec if applicable)")
    st.write("Please do select our preproccesed date columns")
    x_columns = st.multiselect("Select columns for X:", df.columns)

    if not x_columns:
        st.warning("Please select at least one column for X.")
        return

    st.write("Select the target column for Y:")
    y_column = st.selectbox("Select column for Y:", df.columns)

    if not y_column:
        st.warning("Please select a column for Y.")
        return

    df = df.dropna(subset=[y_column])

    X = df[x_columns]
    y = df[y_column]

    # Handle textual data
    textual_columns = X.select_dtypes(include=['object']).columns
    if not textual_columns.empty:
        for col in textual_columns:
            X[col] = X[col].fillna("")  # Fill missing values with empty strings
            vectorizer = TfidfVectorizer()  # You can use any other vectorization method here
            X[col] = vectorizer.fit_transform(X[col])

    numeric_columns = X.select_dtypes(include=['number']).columns
    scaler_option = st.selectbox("Choose a scaler for numerical data:", ('None', 'StandardScaler', 'MinMaxScaler'))

    if scaler_option == 'StandardScaler':
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    elif scaler_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    global X_train,y_train,a,b,c,d
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    a = X_train.shape
    b = y_train.shape

    c=a[0]
    d=b[0]
    
    st.subheader("6.1-Information About Training")
    
    st.write("We have dynamically determined the Architecture of your model using an KNN model trained on our CSV properties vs architecture dataset ")
    lstm = [int(x) for x in output_array_l]
    dense = [int(x) for x in output_array_d]
    model = build_model(lstm,dense,returnseq,bidi,drop)
    model.summary()
    print(model.summary())
   
    st.write("We are going to be training your dataset from our dynamically determined hyperparameters!")
    st.write("The Parameters for your CSV are:")
    st.write("Batch Size",int(batch)) 
    st.write("Epochs",int(epoch))
    st.write("Dropout Value",drop)
    st.write("Bidirectional is",bool(bidi))
    st.write("LSTM Layers",output_array_l)
    st.write("Dense Layers",output_array_d)

    st.write("While we train,here`s a video that should keep you entertained while our algorithm works behind the scenes🎞️!")
    st.write("I mean,who doesn`t like a friends episode?🤔👬🏻👭🏻🫂")
    video_url = "https://www.youtube.com/watch?v=nvzkHGNdtfk&pp=ygUcZnJpZW5kcyBlcGlzb2RlIGZ1bm55IHNjZW5lcw%3D%3D"  # Example YouTube video URL
    st.video(video_url)


    # Train the model
  
    n_steps = 7

# Call the split_data function with X_train and Y_train
    X_train_split, Y_train_split = split_data(X_train, n_steps), split_data(y_train, n_steps)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(epoch), batch_size=int(batch))
    global train_loss
    global val_loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    st.subheader("Training Information➕➖")
    st.write("Final Training loss is-",train_loss[-1])
    st.write("Final Validation loss is-",val_loss[-1])
    st.write("Training losses",train_loss)
    st.write("Validation losses",val_loss)
    # st.write(f"LSTM Model: {model_option}")

    # # Evaluate the model
    # loss, accuracy = model.evaluate(X_test, y_test)
    # st.write(f"Loss: {loss}")
  

# Assuming history is available with the 'loss' and 'val_loss' keys
   
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    ploty()

    model_filename = "model.h5"
    model.save(model_filename)
    st.success(f"Model saved as {model_filename}")

    st.subheader("8.Download the trained model")
    st.download_button(label="Download Model", data=open(model_filename, "rb").read(), file_name=model_filename)

def ploty():
  st.subheader("7.Plotting the loss vs epoch graph")
  epochsi = range(1, len(train_loss) + 1)

  plt.plot(epochsi, train_loss, 'bo', label='Training loss') # 'bo' = blue dots
  plt.plot(epochsi, val_loss, 'r', label='Validation loss') # 'r' = red line   
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  st.write("Yayyyy yipeee!! Now we`re done with processing and training the model!🥳🎉")

# Optionally, you can save the plot or display it
# plt.savefig('loss_plot.png')  # Save the plot as a PNG file
# plt.show()  # Display the plot
#newest
  st.pyplot(plt)


def download_updated_dataset(df):
    st.subheader("9. Download the updated dataset")
    csv_file = df.to_csv(index=False)
    st.download_button("Download CSV", csv_file, "Updated_Dataset.csv", key="csv_download")

def main():
    st.title("LSTM Time Series Dataset Analysis and Model Training App")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.info("File uploaded successfully!")
        df = load_data(uploaded_file)

        if not df.select_dtypes(include=['number']).empty:
            show_correlation(df)
            show_missing_values(df)
            df = handle_missing_values(df)

        df = drop_column(df)
        train_regression_model(df)
  
        download_updated_dataset(df)

if __name__ == "__main__":
    main()
