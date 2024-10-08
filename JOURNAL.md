# Project Journal

## 10/4/2024
I decided to restart this project by using MySQL for the data relevant tasks. I created a table in MySQL and I learned to map the column names and the datatype for each column in to
the table for a more convenient future use. All the work was down in the make_sql_table.py script. I practiced using pandas library by using built in functions to iterate each row
of the feature csv to obtain the Name and the Type. Testing ran successfully.

I also made a deepcopy of the orignal UNSW-NB15_1.csv DataFrame for future data cleaning process. Data cleaning process is started.



## 10/6/2024
Data Cleaning started today, today's focus was on the column 'label', this is a binary column which contains value 0 for benign traffic and 1 for malicious traffic. I did counting and percntage analysis, plotting. The next step is oversampling since the benign traffic is siginificantly more than the malicious traffic.


## 10/8/2024
Oversampling in progress: I separated the column 'label' from the rest of the DataFrame (feature), then I applied SMOTE from imblearn.over_sampling library to resampled both the majority and minority datapoints, which is concatenated to the feature. A new DataFrame is created.
