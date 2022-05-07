## Heart Disease Predictor

### Table of Contents
1. [Introduction](#Introduction)
2. [Getting Started](#GettingStarted)
    1. [Dependency](#Dependencies)
    2. [Installation](#Installation)
    3. [Executing Program](#Execution)
3. [Authors](#Authors)
4. [License](#License)
5. [Acknowledgement](#Acknowledgement)
6. [Screenshots](#Screenshots)

## Introduction <a name=Introduction></a>
### Key Indicators of Heart Disease
2020 annual CDC survey data of 400k adults related to their health status

### What topic does the dataset cover?
According to the CDC, heart disease is one of the leading causes of death for people of most races in the US (African Americans, American Indians and Alaska Natives, and white people). About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicator include diabetic status, obesity (high BMI), not getting enough physical activity or drinking too much alcohol. Detecting and preventing the factors that have the greatest impact on heart disease is very important in healthcare. Computational developments, in turn, allow the application of machine learning methods to detect "patterns" from the data that can predict a patient's condition.

### Where did the dataset come from and what treatments did it undergo?
Originally, the dataset come from the CDC and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents. As the CDC describes: "Established in 1984 with 15 states, BRFSS now collects data in all 50 states as well as the District of Columbia and three U.S. territories. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.". The most recent dataset (as of February 15, 2022) includes data from 2020. It consists of 401,958 rows and 279 columns. The vast majority of columns are questions asked to respondents about their health status, such as "Do you have serious difficulty walking or climbing stairs?" or "Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]". 

### Data 
This data has been downloaded from [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

Here are the columns in the dataset
1. HeartDisease: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
2. BMI: Body Mass Index
3. Smoking: Have you smoked at least 100 cigarettes in your entire life? (Note: 5 packs = 100 cigarettes)
4. AlcoholDrinking: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
5. Stroke: (Ever told) (you had) a stroke?
6. PhysicalHealth: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days)
7. MentalHealth: Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days)
8. DiffWalking: Do you have serious difficulty walking or climbing stairs?
9. Sex: Are you male or female?
10. AgeCategory: Fourteen-level age category
11. Race: American Indian/ Alaskan Native, Asian / Black / White / Other
12. Diabetic: Yes, No, Pre-Diabetic, Yes (During Pregnancy)
13. PhysicalActivity: Yes / No
14. GeneralHealth: Excellent / Very Good / Good / Fair / Poor 
15. SleepTime
16. Asthma: Yes / No
17. KidneyDisease: Yes / No
18. SkinCancer: Yes / No

The project is divided into following sections
* Exploratory Data Analysis
* Machine Learning Pipeline that pre-processes, normalizes and trains model to be used for classification 
* Web application that takes above-mentioned parameters as input and predicts whether an individual has heart disease or not

## Getting Started <a name='GettingStarted'></a>
### Dependencies <a name='Dependencies'></a>
Following packages were used in this project
* Flask
* imblearn
* seaborn
* sklearn
* xgboost

### Installation <a name='Installation'></a>
* Clone this repository using `git clone https://github.com/sumitkumar-00/HeartDisease`
* Install required packages by executing `pipenv install` in the projects root directory

### Executing program <a name='Execution'></a>
1. Run following commands in project's root directory   
   * To execute ML pipeline `pipenv run python model/classifier.py data/heart_2020_cleaned.csv`
   * To run web app execute `pipenv run python run.py` from app's directory
2. Go to http://127.0.0.1:3201 to check out the app 

## Authors <a name='Authors'></a>
. [Sumit Kumar](https://github.com/sumitkumar-00)
## License <a name='License'></a>
Feel free to make changes
## Acknowledgement <a name='Acknowledgement'></a>
I would like to thank Kaggle making this data available
## Screenshots <a name='Screenshots'></a>    

