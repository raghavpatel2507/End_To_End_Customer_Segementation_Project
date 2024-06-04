from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.use('Agg')  


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


def load_and_clean_data(file_path):
    data = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    data["CustomerID"] = data["CustomerID"].astype(str)
    data["Amount"] = data['Quantity'] * data['UnitPrice']
    rfm_m = data.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
    max_date = max(data['InvoiceDate'])
    data['Diff'] = max_date - data['InvoiceDate']
    rfm_p = data.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Calculate IQR
    Q1 = rfm.quantile(0.05).iloc[0]
    Q3 = rfm.quantile(0.95).iloc[0]
    IQR = Q3 - Q1

    # Filter outliers
    rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]
    rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]

    return rfm


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    return rfm, rfm_df_scaled


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    directory = 'uploads'  # Directory to save uploaded files
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file.filename)
    file.save(file_path)
    df = preprocess_data(file_path)[1]
    result_df = model.predict(df)
    df_with_id = preprocess_data(file_path)[0]
    df_with_id['Cluster_Id'] = result_df

    # Generate the images
    sns.stripplot(x='Cluster_Id', y='Amount', data=df_with_id, hue='Cluster_Id')
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id, hue='Cluster_Id')
    freq_img_path = 'static/ClusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id, hue='Cluster_Id')
    recency_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    response = {'amount_img': amount_img_path,
                'freq_img': freq_img_path,
                'recency_img': recency_img_path}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
