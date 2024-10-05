# Step 1: Understanding IDS: N-IDS, H-IDS
"""1. Understanding network intrusion detection system (N-IDS) and host
intrusion detection system (H-IDS):

N-IDS is a solution implemented in the organization's network which monitors the incoming
and outgoing traffic. This solution detects the suspicious traffic in all devices which is
connected to the organization's network.

H-IDS: Think of it as a plan B for H-IDS. It works in certain fields where N-IDS does not work.
It detects the malicious traffic which N-IDS cannot detect. It also works when the host is infected
by malicious traffic and tries to spread it to other devices that are connected to the organization's
network.
"""

import unittest     # provides tools for constructing and running tests
import pandas as pd
import numpy as np
import os


# Load dataset
def load_dataset(csv_file):
    """Load the dataset (UNSW-NB15_1)."""
    df = pd.read_csv(csv_file, low_memory=True)
    return df


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

# Command
"""
check_missing_in_column('UNSW-NB15_1.csv', 48)  # Check missing value in column 48
"""


# Check missing data for all column
def check_missing_for_all_column(csv_file):
    """Return the number of missing values for all columns in the given file csv_file.
    """

    num_of_column = 49
    for i in range(num_of_column + 1):
        check_missing_in_column(csv_file, i)


# Command for check_missing_for_all_column
all_column = (check_missing_for_all_column('UNSW-NB15_1.csv'))
print(all_column)


# Main Commands
load_dataset('UNSW-NB15_1.csv')



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
