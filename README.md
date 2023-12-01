# Sales Prediction Using Python

## Project Overview

This project aims to predict sales based on advertising budgets in different mediums (TV, Radio, Newspaper) using a linear regression model. The project includes data analysis, visualization, model training, evaluation, and a Flask web application for making predictions.

## Data Analysis

- The dataset used for analysis is "Advertising.csv".
- Initial data exploration includes displaying basic information, removing unnecessary columns, handling missing values, and describing the dataset.

## Data Visualization

- Visualizations are created to explore the relationships between advertising budgets and sales.
- A pair plot is generated to visualize the individual relationships between TV, Radio, Newspaper, and Sales.
- A correlation heatmap is used to visualize the correlation between features.

## Script Details

- The script includes functions for data visualization and sales prediction using a linear regression model.
- Model evaluation metrics (MSE, R-squared, MAE) are displayed along with a scatter plot and metrics summary.

**Model Use**: After training, this model is used for prediction. For prediction, run `app.py` file.

## Video Representation
Check out the video representation of the project for a more interactive and engaging overview: [Sales Price Prediction Video](https://youtu.be/NaWSknn7TlY)

## Requirements
Ensure you have the following libraries installed to run the script:

- pandas
- joblib
- seaborn
- matplotlib
- scikit-learn
- flask

Install the required libraries using:

    pip install pandas joblib seaborn matplotlib scikit-learn flask
or

    pip install -r requirements.txt
    
## Usage
To use this project, follow these steps:
1. Ensure you have Python installed on your machine.
2. **For Training:**
   - Clone the repository: `git clone https://github.com/mominurr/oibsip_task5`
   - Install the required libraries: `pip install -r requirements.txt`
   - Run the script: `python sales-predictor.py`
3. **For Prediction:**
   - Run the script: `python app.py`

## Conclusion
This project leverages a linear regression model to predict sales from advertising budgets in various mediums. Through data analysis and visualization, we gained insights into the relationships between TV, Radio, Newspaper, and Sales. The model demonstrates strong performance with low MSE, high R-squared, and minimal MAE. The Flask web application provides a user-friendly interface for making real-time sales predictions. This project serves as a robust foundation for leveraging machine learning in marketing and decision-making processes.

##Author:
[Mominur Rahman](https://github.com/mominurr)
