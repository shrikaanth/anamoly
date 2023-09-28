import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(",".join(result))

def analyze(df):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "\n")
    cols = df.columns.values
    total = float(len(df))
    result_text.insert(tk.END, "{} rows\n".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count > 100:
            result_text.insert(tk.END, "** {}:{} ({}%)\n".format(col, unique_count, int(((unique_count) / total) * 100)))
        else:
            result_text.insert(tk.END, "** {}:{}".format(col, expand_categories(df[col])) + "\n")

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def load_data():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        analyze(df)
        result_text.insert(tk.END, "\nData loaded successfully!\n")

def process_data():
    global df
    if df is not None:
        df1 = df.copy()
        encode_numeric_zscore(df, 'duration')
        encode_text_dummy(df, 'protocol_type')
        encode_text_dummy(df, 'service')
        encode_text_dummy(df, 'flag')
        encode_numeric_zscore(df, 'src_bytes')
        encode_numeric_zscore(df, 'dst_bytes')
        encode_text_dummy(df, 'land')
        encode_numeric_zscore(df, 'wrong_fragment')
        encode_numeric_zscore(df, 'urgent')
        encode_numeric_zscore(df, 'hot')
        encode_numeric_zscore(df, 'num_failed_logins')
        encode_text_dummy(df, 'logged_in')
        encode_numeric_zscore(df, 'num_compromised')
        encode_numeric_zscore(df, 'root_shell')
        encode_numeric_zscore(df, 'su_attempted')
        encode_numeric_zscore(df, 'num_root')
        encode_numeric_zscore(df, 'num_file_creations')
        encode_numeric_zscore(df, 'num_shells')
        encode_numeric_zscore(df, 'num_access_files')
        encode_numeric_zscore(df, 'num_outbound_cmds')
        encode_text_dummy(df, 'is_host_login')
        encode_text_dummy(df, 'is_guest_login')
        encode_numeric_zscore(df, 'count')
        encode_numeric_zscore(df, 'srv_count')
        encode_numeric_zscore(df, 'serror_rate')
        encode_numeric_zscore(df, 'srv_serror_rate')
        encode_numeric_zscore(df, 'rerror_rate')
        encode_numeric_zscore(df, 'srv_rerror_rate')
        encode_numeric_zscore(df, 'same_srv_rate')
        encode_numeric_zscore(df, 'diff_srv_rate')
        encode_numeric_zscore(df, 'srv_diff_host_rate')
        encode_numeric_zscore(df, 'dst_host_count')
        encode_numeric_zscore(df, 'dst_host_srv_count')
        encode_numeric_zscore(df, 'dst_host_same_srv_rate')
        encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
        encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
        encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
        encode_numeric_zscore(df, 'dst_host_serror_rate')
        encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
        encode_numeric_zscore(df, 'dst_host_rerror_rate')
        encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')
        df2 = df.copy()
        df.dropna(inplace=True, axis=1)
        result_text.insert(tk.END, "\nData processed successfully!\n")
    else:
        result_text.insert(tk.END, "\nPlease load the data first!\n")

def train_model():
    global df
    if df is not None:
        x_columns = df.columns.drop('labels')
        x = df[x_columns].values
        dummies = pd.get_dummies(df['labels']) 
        outcomes = dummies.columns
        num_classes = len(outcomes)
        y = dummies.values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        model = Sequential()
        model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
        model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=2, epochs=1000)
        pred = model.predict(x_test)
        pred = np.argmax(pred, axis=1)
        y_eval = np.argmax(y_test, axis=1)
        score = metrics.accuracy_score(y_eval, pred)

        # Confusion Matrix
        cm = confusion_matrix(y_eval, pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        classes = range(len(outcomes))
        plt.xticks(classes, outcomes, rotation=45)
        plt.yticks(classes, outcomes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        # Validation Score
        result_text.insert(tk.END, "\nValidation score: {}\n".format(score))
    else:
        result_text.insert(tk.END, "\nPlease load and process the data first!\n")

def train_rf_model():
    global df
    if df is not None:
        x_columns = df.columns.drop('labels')
        x = df[x_columns].values
        dummies = pd.get_dummies(df['labels']) 
        outcomes = dummies.columns
        num_classes = len(outcomes)
        y = dummies.values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        model1 = RandomForestClassifier(n_estimators=20)
        model1.fit(x_train, y_train)
        rf_score = model1.score(x_test, y_test)

        # Predict using the model
        pred = model1.predict(x_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        classes = range(len(outcomes))
        plt.xticks(classes, outcomes, rotation=45)
        plt.yticks(classes, outcomes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        # Random Forest Validation Score
        result_text.insert(tk.END, "\nRandom Forest Validation Score: {}\n".format(rf_score))
    else:
        result_text.insert(tk.END, "\nPlease load and process the data first!\n")

def clear_result():
    result_text.delete(1.0, tk.END)

def main():
    global df, result_text

    root = tk.Tk()
    root.title("Anomaly Detection")

    # Load Data Button
    load_data_button = tk.Button(root, text="Load Data", command=load_data)
    load_data_button.pack(pady=10)

    # Process Data Button
    process_data_button = tk.Button(root, text="Process Data", command=process_data)
    process_data_button.pack(pady=10)

    # Train Model Button
    train_model_button = tk.Button(root, text="Train Model", command=train_model)
    train_model_button.pack(pady=10)

    # Train Random Forest Model Button
    train_rf_model_button = tk.Button(root, text="Train RF Model", command=train_rf_model)
    train_rf_model_button.pack(pady=10)

    # Clear Result Button
    clear_result_button = tk.Button(root, text="Clear Result", command=clear_result)
    clear_result_button.pack(pady=10)

    # Result Textbox
    result_text = tk.Text(root, height=10, width=60)
    result_text.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    df = None
    result_text = None
    main()
