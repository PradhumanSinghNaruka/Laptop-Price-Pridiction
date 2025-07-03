from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        company = data['company']
        typename = data['typename']
        ram = int(data['ram'])
        weight = float(data['weight'])
        touchscreen = 1 if data['touchscreen'] == 'Yes' else 0
        ips = 1 if data['ips'] == 'Yes' else 0
        ppi = float(data['ppi'])
        cpu = data['cpu']
        hdd = int(data['hdd'])
        ssd = int(data['ssd'])
        gpu = data['gpu']
        os = data['os']

        input_data = np.array([[company, typename, ram, weight,
                                touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]])
        predicted = pipeline.predict(input_data)[0]
        final_price = np.exp(predicted)

        return render_template('index.html', prediction_text=f"💻 Predicted Laptop Price: ₹{int(final_price):,}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
