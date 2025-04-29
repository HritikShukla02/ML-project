from flask import Flask, render_template, request
import json # Import json library to pretty-print the dictionary
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handles both displaying the form (GET) and processing form submissions (POST).
    """
  
        # Re-render the same template, passing the response data back
    return render_template('home.html')

    # --- Handle GET Request ---
    # If it's a GET request, just display the form without any response data
    return render_template('home.html', response_data=None)

@app.route("/predictdata", methods=['GET', "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
        gender=request.form.get('option1'),
        race_ethnicity=request.form.get('option2'),
        parental_level_of_education=request.form.get('option3'),
        lunch=request.form.get('option4'),
        test_preparation_course=request.form.get('option5'),
        reading_score=request.form.get('num1', 0),
        writing_score=request.form.get('num2', 0)
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        pred_pipeline = PredictPipeline()
        result = pred_pipeline.predict(pred_df)

        return render_template('home.html', result=result)
if __name__ == '__main__':
    # Run the Flask app in debug mode (reload on changes, show detailed errors)
    # Turn off debug mode for production environments
    app.run(debug=True)