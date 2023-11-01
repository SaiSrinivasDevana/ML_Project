# Heart Disease Prediction 
## Introduction About the Data :

The dataset: The goal is to predict the risk of an individual being subjected to heart disease (Classification Analysis).

There are 13 independent variables:

- age : Represents age of an individual
- sex : is an indicator variable takes 2 values ( 1 for Male, 0 for Female)
- chest pain type : Represents the type of chest pain experienced by individual which takes 4 values (Myocardial Infarction/Angina Pectoris/Other Cardiac Chest Pain/Non-cardiac Chest Pain) 
- trestbps : Represents the blood pressure level
- chol : Represents serum cholestoral in mg/dl
- fbs: Is indicator variable showing whether fasting blood sugar > 120 mg/dl or not
- restecg : Represents resting electrocardiographic results (values 0,1,2)
- thalach : Represents maximum heart rate achieved.
- exang : It is an indicator variable indicating whether angina is due to exercise induced angina or not
- oldpeak = ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
- ca: number of major vessels (0-3) colored by flourosopy
- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
- Target variable:

target: The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

- Dataset Source Link : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

AWS Deployment Link :
- AWS Elastic Beanstalk link : https://us-east-1.console.aws.amazon.com/codesuite/codepipeline/pipelines/heart_disease_prediction/view?region=us-east-1


Approach for the project

Initially Exploratory data analysis is performed to look at the initial raw data and figured out the steps that has to be performed in the pipeline before applying any model on top of it. 

Data Ingestion :

In Data Ingestion phase the data is first read as csv.
Then the data is split into training and testing and saved as csv file.
Data Transformation :

In this phase a Pipeline is created.
- Initially for droping duplicate features DropDuplicateFeatures function from feature engine is used.

- for Numeric Variables first MeanMedianImputer is applied with strategy median , then log transformation is performed followed by which Standard Scaling is performed on numeric data.
- for Categorical Variables Categorical Imputer is applied with missing strategy, then rare label encoding performed for columns which contain categories with less than 5% of data, after this one hot encoding is performed.
- This preprocessor is saved as pickle file.
  
Model Training :

In this phase different models with different hyper parameter combinations were tested . The best model found was Random Forest regressor.
This model is saved as pickle file.

Prediction Pipeline :

This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.
Flask App creation :

Flask app is created with User Interface to predict the gemstone prices inside a Web Application.
Exploratory Data Analysis Notebook
Link :[EDANotebook] [https://github.com/SaiSrinivasDevana/ML_Project/blob/main/notebook/EDA.ipynb]

Model Training Approach Notebook
Link : Model Training Notebook



