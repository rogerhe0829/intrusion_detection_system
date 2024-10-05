import pandas as pd

# Excel sheet info
path = '/Users/rogerhe0829/Desktop/projects/IDS/NUSW-NB15_features.csv'

# Read the Excel spreadsheet
df = pd.read_csv(path)

# Create table
table_name = 'UNSW-NB15_1'
columns = [] # A list of column names

# Iterate through the DataFrame rows to construct the column definitions
for index, row in df.iterrows():
    column_name = row['Name']
    data_type = row['Type ']
    columns.append(f"{column_name} {data_type}") # append a string contains both column_name and data_type with space between

# Join the columns to MySQL
sql_table = ',\n'.join(columns)
sql_make_table_command = f'CREATE TABLE {table_name} (\n {sql_table} \n);'
print(sql_make_table_command)