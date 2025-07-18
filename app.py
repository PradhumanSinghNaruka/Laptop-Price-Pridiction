from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
model = pickle.load(open('pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # FORM FIELDS EXACT SAME
        company = request.form['Company']
        typename = request.form['TypeName']
        ram = int(request.form['Ram'])
        weight = float(request.form['Weight'])
        touchscreen = int(request.form['Touchscreen'])
        ips = int(request.form['Ips'])
        ppi = float(request.form['Ppi'])
        cpu_brand = request.form['CpuBrand']
        hdd = int(request.form['HDD'])
        ssd = int(request.form['SSD'])
        gpu_brand = request.form['GpuBrand']
        os = request.form['Os']

        # ✅ FINAL INPUT DATA ORDER MATCHING X_train:
        input_data = np.array([[company, typename, ram, weight,
                                touchscreen, ips, ppi, cpu_brand,
                                hdd, ssd, gpu_brand, os]])
        print("INPUT DATA", input_data)
        log_price = model.predict(input_data)[0]
        predicted_price = np.exp(log_price)
        
        return render_template('index.html',
                               prediction_text=f"Predicted Laptop Price: ₹{round(predicted_price)}")

if __name__ == "__main__":
    app.run(debug=True)
# ye import zarur karo

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             company = request.form['Company']
#             typename = request.form['TypeName']
#             ram = int(request.form['Ram'])
#             weight = float(request.form['Weight'])
#             touchscreen = int(request.form['Touchscreen'])
#             ips = int(request.form['Ips'])
#             ppi = float(request.form['Ppi'])
#             cpu_brand = request.form['CpuBrand']
#             hdd = int(request.form['HDD'])
#             ssd = int(request.form['SSD'])
#             gpu_brand = request.form['GpuBrand']
#             os = request.form['Os']

#             # ✅ DataFrame banana hoga
#             input_dict = {
#                 'Company': [company],
#                 'TypeName': [typename],
#                 'Ram': [ram],
#                 'Weight': [weight],
#                 'Touchscreen': [touchscreen],
#                 'Ips': [ips],
#                 'Ppi': [ppi],
#                 'Cpu brand': [cpu_brand],
#                 'HDD': [hdd],
#                 'SSD': [ssd],
#                 'Gpu Brand': [gpu_brand],
#                 'OpSys': [os]
#             }

#             return ("Predicted Laptop Price is between : ₹45,000 to 55,000");
