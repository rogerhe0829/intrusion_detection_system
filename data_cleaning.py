import unittest     # provides tools for constructing and running tests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


# Load dataset UNSW-NB15_1
def load_dataset(csv_file):
    """Load the dataset (UNSW-NB15_1)."""
    df = pd.read_csv(csv_file, low_memory = False)
    return df


# From ErrorMessage I learned that column [1,3,47] have mixed data types.
# Normalize Data Types of the listed columns
def rearrange_type(csv_file):
    df = load_dataset(csv_file)

    # From feature csv, column 1 and 3 should be  treated as categorical (nominal)
    df.iloc[:, 1] = df.iloc[:, 1].astype('category')  # Convert to categorical
    df.iloc[:, 3] = df.iloc[:, 3].astype('category')  # Convert to categorical

    # From feature csv, column 47 should be  treated as integer
    df.iloc[:, 47] = pd.to_numeric(df.iloc[:, 47], errors='coerce').fillna(0).astype(int)  # Convert to integer, filling NaNs with 0
    return df


# Command
rearrange_type('UNSW-NB15_1.csv')

# Assign each column to its name
def assign_column_names(feature_file):
    """Return arrays of columns from feature CSV."""
    df = load_dataset(feature_file)  # Correctly load from feature_file
    column_num = df.iloc[:, 0].astype(int).values  # Column numbers
    name = df.iloc[:, 1].values  # Column names
    data_type = df.iloc[:, 2].values  # Data types
    description = df.iloc[:, 3].values  # Descriptions
    return column_num, name, data_type, description


def check_missing_in_row(csv_file, row_num):
    """Return number of missing values in a given row row_num of the file csv_file.
    """

    # load data
    df = load_dataset(csv_file)
    # find missing value in first column
    num_of_missing_value = df.iloc[row_num -1].isnull().sum()
    print('There are {} number of missing value in row {}.'.format(num_of_missing_value, row_num))


def check_missing_in_column(csv_file, column_num):
    """Return number of missing values in a given column column_num of the file csv_file.
    """

    # load data
    df = load_dataset(csv_file)

    # find missing value in first column
    num_of_missing_value = df.iloc[:,column_num - 1].isnull().sum()
    print('There are {} number of missing value in column {} out of {}.'.format(num_of_missing_value, column_num, len(df)))


# Check missing data for all column
def check_missing_for_all_column(csv_file):
    """Return the number of missing values for all columns in the given file csv_file.
    """

    num_of_column = 49
    for i in range(num_of_column + 1):
        check_missing_in_column(csv_file, i)


# Check attacking label
def label_count(csv_file):
    # load dataset
    df = load_dataset(csv_file)
    # function
    label_count = df.iloc[:, 48].value_counts()
    # count total malicious traffic
    malicious_count = label_count.get(1, 0)
    # count total benign traffic
    benign_count = label_count.get(0, 0)
    return malicious_count, benign_count


# Display label analysis
def label_count_print(malicious_count, benign_count, csv_file):
    print(f'There are {malicious_count} malicious traffic in {csv_file}.')
    print(f'There are {benign_count} benign traffic in {csv_file}.')


# show percentage
def label_percentage(csv_file):
    df = load_dataset(csv_file)
    malicious_count, benign_count = label_count(csv_file)
    malicious_percent = malicious_count / len(df.iloc[:, 48]) * 100
    benign_percent = benign_count / len(df.iloc[:, 48]) * 100
    print(f'The percentage of malicious traffic is {round(malicious_percent,2)}%')
    print(f'The percentage of benign traffic is {round(benign_percent, 2)}%')


# show pie chart
def label_barplot(csv_file):
    # Get counts from label_count function
    malicious_count, benign_count = label_count(csv_file)
    # Display label analysis
    label_count_print(malicious_count, benign_count, csv_file)
    # labels and data for plotting
    labels = ['Benign (0)', 'Malicious (1)']
    counts = [benign_count, malicious_count]

    # Plotting
    bars = plt.bar(labels, counts,
                   color=['blue', 'red'],
                   width = 0.5,
                   bottom = 0,
                   label = ['Benign (0)', 'Malicious (1)']
                   )
    # Add count values on top of the bars
    for bar in bars:
        count = bar.get_height()  # Get the height of the bar
        plt.text(bar.get_x() + bar.get_width() / 2,
                 count,
                 int(count),
                 ha='center',
                 va='bottom'    # Position the count text on top of the bar
                 )
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Count of Benign and Malicious Traffic')
    plt.legend()
    plt.show()


#Command
label_barplot('UNSW-NB15_1.csv')
label_percentage('UNSW-NB15_1.csv')



# Since there exists a significant class imbalance, I will apply oversampling method to
# increase the instance of malicious traffic. This helps the ML model to know more about
# the malicious traffic without losing significant information of the benign traffic.

# Import train_test_split to split the data
# from sklearn.model_selection import train_test_split
# We use SMOTE

# load data
df = load_dataset('UNSW-NB15_1.csv')
df = rearrange_type('UNSW-NB15_1.csv')


# Separate features and labels
X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, 48]   # The 49th column (label)

# initialize SMOTE
smote = SMOTE(sampling_strategy='auto', random_state = 42) # 42 is a common seed
# Apply SMOTE
X_smote, y_smote = smote.fit_resample(X, y) #X_smote: feature, y_smote: oversampled 'label' column

# Concatenate the resampled features and labels into a new DataFrame
smote_data = pd.concat([pd.DataFrame(X_smote), pd.Series(y_smote, name='label')], axis=1)

# concatenate to original DataFrame

print(y_smote.value_counts())




# For model training (optional)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix




# Testing
class TestAssignColumnNames(unittest.TestCase):  # class contains test methods

    @classmethod
    def setUpClass(cls):  # class-level set up (one for each test class) ; cls: class itself
        # Create a temporary test CSV file
        cls.test_csv = 'test_NUSW-NB15_features.csv'
        data = """Column Number,Name,Data Type,Description
        1,srcip,string,Source IP address
        2,dstip,string,Destination IP address
        3,srcport,int,Source port number
        4,dstport,int,Destination port number"""

        with open(cls.test_csv, 'w') as f:  # open a temporary file
            f.write(data)  # write CSV in temporary file


    def test_assign_column_names(self):
        """Test the assign_column_names function."""
        column_num, name, data_type, description = assign_column_names(self.test_csv)

        # Test the results
        expected_column_num = np.array([1, 2, 3, 4])
        expected_name = np.array(['srcip', 'dstip', 'srcport', 'dstport'])
        expected_data_type = np.array(['string', 'string', 'int', 'int'])
        expected_description = np.array([
            'Source IP address',
            'Destination IP address',
            'Source port number',
            'Destination port number'
        ])

        np.testing.assert_array_equal(column_num, expected_column_num)
        np.testing.assert_array_equal(name, expected_name)
        np.testing.assert_array_equal(data_type, expected_data_type)
        np.testing.assert_array_equal(description, expected_description)


    @classmethod
    def tearDownClass(cls):  # this class method runs once after all tests have been executed
        # Remove the test CSV file after tests
        if os.path.exists(cls.test_csv):
            os.remove(cls.test_csv)  # remove sample CSV to keep the test environment clean


if __name__ == '__main__':
    unittest.main()
