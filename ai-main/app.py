import os
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from predict import preprocess_data, encode_categorical, predict_anomalies, model, scaler
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            df = pd.read_csv(file)
            new_data_processed = preprocess_data(df)
            cat_columns = ['LOCATION', 'IP ADDRESS']
            categories = [
                new_data_processed['LOCATION'].unique(),
                new_data_processed['IP ADDRESS'].unique()
            ]
            new_data_encoded = encode_categorical(new_data_processed, cat_columns, categories)
            anomaly_scores = predict_anomalies(model, new_data_encoded, scaler)
            results = [(i + 1, score, 'Fraudulent' if score > 0.5 else 'Normal') for i, score in enumerate(anomaly_scores)]
            
            # Generate line chart and save as image
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(anomaly_scores) + 1), anomaly_scores, marker='o', linestyle='-')
            plt.title('Anomaly Scores Line Chart')
            plt.xlabel('Transaction')
            plt.ylabel('Anomaly Score')
            plt.grid(True)
            image_path = os.path.join('static', 'anomaly_scores_line_chart.png')
            plt.savefig(image_path)
            plt.close()

            return render_template('results.html', results=results, image_path=image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
