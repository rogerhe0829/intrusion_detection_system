# Project Journal

OBJ: The goal of this project is to build an IDS based on dataset UNSW-NB15_1.csv. Basic structure: understanding the dataset => data cleaning => feature engineering => select ML algorithm => building ML model => testing.

## 10/4/2024
Project Starts.

## 10/6/2024
Data Cleaning started today, today's focus was on the column 'label', this is a binary column which contains value 0 for benign traffic and 1 for malicious traffic. I did counting and percntage analysis, plotting. The next step is oversampling since the benign traffic is siginificantly more than the malicious traffic.


## 10/8/2024
Oversampling in progress: I separated the column 'label' from the rest of the DataFrame (feature), then I applied SMOTE from imblearn.over_sampling library to resampled both the majority and minority datapoints, which is concatenated to the feature. A new DataFrame is created.
