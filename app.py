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
        # get data from form using exact field names
        Company = request.form['Company']
        TypeName = request.form['TypeName']
        Ram = int(request.form['Ram'])
        Weight = float(request.form['Weight'])
        Touchscreen = request.form['Touchscreen']
        Ips = request.form['Ips']
        Ppi = float(request.form['Ppi'])
        Cpu = request.form['Cpu brand']
        HDD = int(request.form['HDD'])
        SSD = int(request.form['SSD'])
        Gpu = request.form['Gpu Brand']
        Os = request.form['Os']

        # Order matters! Match this to training order
        final_data = np.array([Company, TypeName, Ram, Weight, Touchscreen, Ips, Ppi, Cpu, HDD, SSD, Gpu, Os]).reshape(1, -1)

        prediction = pipeline.predict(final_data)[0]
        return render_template('index.html', prediction_text=f"Predicted Price: ₹{int(prediction)}")
    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
