from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
MODEL = joblib.load('sales-predictor.pkl')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input values from the form
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        # Make a prediction using the loaded model
        input_data = np.array([[tv, radio, newspaper]])
        prediction = MODEL.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)
    
    # Render the initial input form
    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
