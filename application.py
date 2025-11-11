from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car_df = pd.read_csv('Cleaned Car.csv')

# Extract unique values for dropdowns
companies = sorted(car_df['company'].unique())
car_names = sorted(car_df['name'].unique())
fuel_types = sorted(car_df['fuel_type'].unique())
years = sorted(car_df['year'].unique(), reverse=True)  # latest years first

@app.route("/", methods=['GET', 'POST'])
def index():
    predicted_price = None

    if request.method == 'POST':
        name = request.form.get('name')
        company = request.form.get('company')
        year = int(request.form.get('year'))
        kms_driven = int(request.form.get('kms_driven'))
        fuel_type = request.form.get('fuel_type')

        # Create input DataFrame for prediction
        input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Predict price
        predicted_price = round(model.predict(input_data)[0], 2)

    return render_template("index.html",
                           companies=companies,
                           car_names=car_names,
                           fuel_types=fuel_types,
                           years=years,
                           predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
