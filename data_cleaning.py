import pandas as pd

# OBJ: clean data for the data insertion to the machine learning model.


# Make a DataFrame copy
df = pd.read_csv('UNSW-NB15_1.csv')
copy = df.copy()

# Start Data Analysis
#Check missing data
