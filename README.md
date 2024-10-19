# Golfers Performaces and Points Prediction

This project focuses on predicting a player's average score and points using historical data from the PGA Tour. We will apply data wrangling, machine learning, and visualization techniques in Python to explore insights and make predictions. To predict the golfer's average score and points, I used multiple regression methods such as Linear Regression, Random forest Regressor, and Gradient Boosting Regressor. I found that I had the best performance with the Linear Regression method when predicting average scores and the Random Forest method when predicting points after fine-tuning.


## Skills

- Data cleaning
- Data wrangling
- Data visualization
- Linear Regression 
- Random Forest Regressor
- Gradient Boosting Regressor
- Fine tunning

## Installation and Usage

For this project you will need to install Jupyter notebook.

For local installation make sure to have [pip installed](https://pip.pypa.io/en/stable/installation/) and run:
```bash
  pip install notebook
```

To run the notebook in local installation, launch with:
```bash
  jupyter notebook
```
    
## Data

### Data Source
[PGA Tour Data](https://www.kaggle.com/datasets/jmpark746/pga-tour-data-2010-2018)

### Data Description

**Player Name:** Name of the golfer. \
**Rounds:** Number of rounds played. \
**Fairway Percentage:** Percentage of times the player hits the fairway off the tee. \
**Year:** The season year. \
**Avg Distance:** Average driving distance. \
**GIR (Greens in Regulation):** Percentage of greens hit in regulation. \
**Average Putts:** Average number of putts per round. \
**Average Scrambling:** Percentage of successful scrambling (making par or better after -missing the green in regulation). \
**Average Score:** Average score per round (dependent variable). \
**Points:** FedEx Cup points (another dependent variable). \
**SG (Strokes Gained):** Metrics for putting, total, off-the-tee, approach, and around-the-green. \
**Wins, Top 10:** Count of wins and top 10 finishes. \
**Money:** Total earnings in a season.


## Python Libraries Installation

This project requires the use of Python libraries pandas, seaborn, and sklearn. To install these libraries, run the following code in your notebook:

```python 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
```

## Exploratory Data Analysis:

![image alt](https://github.com/Rwang0316/Golfer-Performance-and-Points-Prediction/blob/main/Media/heatmap.png)

The heatmap above displays the correlations between the selected features and the target variables (Average Score and Points).

High Negative Correlation with Average Score: Features like SG: Total (-0.95), SG: Putts (-0.88), and GIR (-0.78) show strong negative correlations with Average Score, indicating that as these values increase, a player’s score tends to decrease (which is desirable in golf).

Strong Positive Correlation with Points: SG: Total (0.77), SG: ARG (0.63), and Fairway Percentage (0.49) are positively correlated with Points, showing that improved performance in these areas typically results in higher overall points.

Moderate Correlations: Features such as Driving Distance and Scrambling show moderate correlations, contributing to a player's overall performance but not as strongly as the strokes gained metrics.

Surprises: Golfers are often reminded by their coach how importance it is to hit the fairway in order to improve their score, however Fairway Percentage (-0.039) didn't have a strong or even moderate correlation with average scores.



## Result and Evaluation

In this project, we utilized the [Kaggle PGA Tour](https://www.kaggle.com/datasets/jmpark746/pga-tour-data-2010-2018) dataset to predict Points and Average Score of golfers using several machine learning models. After selecting relevant features based on performance metrics like Strokes Gained, Fairway Percentage, and Greens in Regulation (GIR), we trained and fine-tuned the models to optimize their performance.

***Model Evaluation for Average Score Prediction***
| Model | MSE     | R^2 |
| :-------- | :------- | :------------------------- |
| Linear Regression|0.0366|0.9300|
| Random Forest|0.0501|0.9042|
| Gradient Boosting|0.0537|0.8974|
| Support Vector Regressor|0.0603|0.8847|


***Model Evaluation for Points Prediction***
| Model | MSE     | R^2 | R^2 after Fine Tuning|
| :-------- | :------- | :------------------------- | :----------------|
| Linear Regression|98907.3588|0.5902|N/A|
| Random Forest|91361.9298|0.6215|0.6423|
| Gradient Boosting|82019.3557|0.6602|0.6032|
| Support Vector Regressor|232346.5447|0.0374|0.6189|


Key findings include:

- Linear Regression performed the best for predicting Average Score, with an R² score of 0.930.
- Gradient Boosting outperformed other models for predicting Points, achieving an R² score of 0.6423 after fine-tuning.
- Random Forest and Support Vector Regressor (SVR) also performed well but required further tuning to reach optimal results.

The use of GridSearchCV allowed us to fine-tune hyperparameters and enhance model accuracy, showcasing how thoughtful hyperparameter tuning can improve model performance. By evaluating different models and considering both MSE and R² as evaluation metrics, we were able to select the most effective models for predicting golf scores and player points.
## Future work

After performing some simple regression models on predicting the average score and points of players, I'd like to revisit this project with a deeper understanding of other regression modes to predict my target variables more accurately. I could also create new features by combining existing ones, such as an efficiency metric that combines driving accuracy with distance and temporal features like seasonal trends to better understand the correlation between my feature and target variables. Inspired by this project, I could also use this dataset to solve different types of ML problems such as a classification one to predict if a player will win a tournament in a certain year. This will introduce me to more ML models and help me become a better data scientist.

**Thanks for reading!**
