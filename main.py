from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for HomePage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            car_name=request.form.get('car_name'),
            brand=request.form.get('brand'),
            model=request.form.get('model'),
            vehicle_age=int(request.form.get('vehicle_age')),
            km_driven=float(request.form.get('km_driven')),
            seller_type=request.form.get('seller_type'),
            fuel_type=request.form.get('fuel_type'),
            transmission_type=request.form.get('transmission_type'),
            mileage=float(request.form.get('mileage')),
            engine=int(request.form.get('engine')),
            max_power=float(request.form.get('max_power')),
            seats=int(request.form.get('seats'))
        )
        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
