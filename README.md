# SkySaver: Airline Price Prediction App

## Overview
SkySaver is a machine learning model designed to predict airline ticket prices. By analyzing various travel factors such as airline, departure and arrival times, stops, and travel class, this project aims to provide insights into flight pricing trends. Additionally, SkySaver includes a **Streamlit-powered web app** that allows users to input travel details and get estimated ticket prices.

## Dataset
The dataset used in this project is **Indian Airlines.csv**, which contains airline ticket details, including:
- **airline**: Airline name
- **flight**: Flight number
- **source_city**: City of departure
- **departure_time**: Time of departure
- **stops**: Number of stops (zero, one, two or more)
- **arrival_time**: Time of arrival
- **destination_city**: City of arrival
- **class**: Travel class (Economy/Business)
- **duration**: Flight duration in hours
- **days_left**: Days left for travel
- **price**: Ticket price

## Methodology
The model is built using **Regression Models**, as they provided the best results in terms of accuracy and generalization. The key steps involved in the project are:
1. **Data Preprocessing**
   - Handling missing values
   - Removing duplicates
   - Feature selection and scaling
2. **Exploratory Data Analysis (EDA)**
   - Identifying patterns and correlations
   - Visualizing key metrics
3. **Model Training & Evaluation**
   - Splitting data into training and test sets
   - Training various regression models (Linear Regression, Ridge Regression, Lasso Regression,RandomForest etc)
   - Evaluating performance using metrics like R-squared, Mean Absolute Error (MAE), and Mean Squared Error (MSE)

## Results
- **Regression Models** provided strong predictive capabilities.
- Key travel indicators strongly influenced price predictions.
- The model can be used as a decision-support tool for passengers and airlines.

## Airline Price Prediction App
SkySaver also includes a **Streamlit web application** that predicts airline ticket prices based on user inputs such as airline, source city, destination city, travel class, departure and arrival times, duration, and days left for travel.

### Running the Streamlit App
To run the web application, install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn pickle-mixin
```
Then, navigate to the project directory and launch the app:
```bash
streamlit run App.py
```
This will open a browser window where users can input flight details and get an estimated ticket price.

## Installation & Requirements
To run this project, install the required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Running the Model
1. Clone the repository:
```bash
git clone https://github.com/nithyashakthi/SkySaver.git
```
2. Navigate to the project directory:
```bash
cd SkySaver
```
3. Run the Python script or Jupyter Notebook:
```bash
python skysaver_model.py
# OR
jupyter notebook skysaver_notebook.ipynb
```

## Future Enhancements
- Incorporating deep learning models for better accuracy
- Expanding dataset with global airline data
- Deploying the Streamlit app online for wider accessibility
- Improving model speed and efficiency


## Contribution
Feel free to contribute by submitting pull requests or opening issues.

## Contact
For any questions, reach out via [LinkedIn](https://www.linkedin.com/in/nithya-shakthi-b-90135a31b/) or email: bnihya808@gmail.com.

