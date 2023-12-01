# project name : Sales Prediction Using Python
# author : Mominur Rahman
# Date : 01-11-2022
# version : 1.0
# GitHub Repo: https://github.com/mominurr/oibsip_task5
# LinkedIn: https://www.linkedin.com/in/mominur-rahman-145461203/


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


# This function is used to visualize the relationship between the features and the target variable.
def data_visualization(df):

    # Visualize the relationship between the features and the target
    sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=6, kind='reg')
    # sns.pairplot(df)
    plt.savefig('pairplot.png')
    plt.show()

    # Visualize the relationship between the features and the target using a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('heatmap.png')
    plt.show()



# This function is used to predict sales using linear regression model. After training the model, it is used to predict
# the sales for a given value of TV, Radio, and Newspaper.
def sale_prediction():
    # Load the dataset
    df = pd.read_csv("Advertising.csv")

    # Display basic dataset information
    print("\nFirst 5 rows of the dataset: \n")
    print(df.head())

    print("\n\nLast 5 rows of the dataset: \n")
    print(df.tail())

    print("\n\nShape of the dataset: \n")
    print(df.shape)

    print("\n\nData types of the dataset: \n")
    print(df.dtypes)

    print("\n\nInformation about the dataset: \n")
    print(df.info())

    # remove unnecessary columns
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    print("\n\nAfter removing the unnecessary columns.First 5 rows of the dataset: \n")
    print(df.head())

    print("\n\nNull or missing values in the dataset: \n")
    print(df.isnull().sum())

    print("\n\nDescribe the dataset: \n")
    print(df.describe())

    # visualize the relationship between the features and the target.
    data_visualization(df)

    # Split the dataset into training and testing sets. Here we will use 80% of the data for training and 20% for testing.
    X=df.drop('Sales', axis=1).values
    y=df['Sales'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Create a linear regression model and fit it to the training data.
    MODEL=LinearRegression()
    MODEL.fit(X_train, y_train)

    y_pred=MODEL.predict(X_test)

    # Evaluate the model using the testing data
    MSE=mean_squared_error(y_test, y_pred)
    R2=r2_score(y_test, y_pred)
    MAE=mean_absolute_error(y_test, y_pred)
    
    # show the model evaluation result and regression line using matplotlib
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Subplot 1: Scatter plot with regression line
    ax1.scatter(y_test, y_pred)
    # ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Regression Line')
    ax1.set_title('Actual vs. Predicted Sales')
    ax1.set_xlabel('Actual Sales')
    ax1.set_ylabel('Predicted Sales')
    ax1.legend()


    # Subplot 2: Metrics
    metrics_text = f'MSE: {MSE:.2f}\nR-squared: {R2:.2f}\nMAE: {MAE:.2f}'
    ax2.text(0.5, 0.5, metrics_text, verticalalignment='center', horizontalalignment='center', color='blue', fontsize=14)
    ax2.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model-evaluation-report.png')
    plt.show()
    
    print(metrics_text)

    # Save the model
    joblib.dump(MODEL, 'sales-predictor.pkl')   



# here code executed start

if __name__ == "__main__":
    sale_prediction()














