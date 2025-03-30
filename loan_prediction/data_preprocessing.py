import pandas as pd
import numpy as np

class DataProcessor:
    """
    A class for preprocessing loan prediction data.
    This class handles the removal of unnecessary columns, 
    and filling missing values with the median of numerical columns.

    Example usage:
    
    # Example dataset
    data = {
        'ID': [1, 2, 3, 4],
        'Age': [25, 30, 30, 35],
        'Income': [50000, 30000, 55000, 60000],
        'ZIP.Code': [12345, 23456, 34567, 45678],
        'Loan_Status': ['Approved', 'Denied', 'Approved', 'Denied']
    }

    # Creating a DataFrame from the example data
    df = pd.DataFrame(data)

    # Initialize the DataProcessor class with the raw data
    processor = DataProcessor(df)

    # Perform preprocessing
    processed_data = processor.preprocess()

    # Output the processed data
    print(processed_data)
    """

    def __init__(self, data):
        """
        Initializes the DataProcessor class.

        Parameters:
        data (pd.DataFrame): The raw data that needs preprocessing.
        """
        self.data = data.copy()
    
    def remove_unnecessary_columns(self):
        """Removes unnecessary columns such as 'ID' and 'ZIP.Code' if they exist."""
        if 'ID' in self.data.columns:
            self.data = self.data.drop('ID', axis=1)
        if 'ZIP.Code' in self.data.columns:
            self.data = self.data.drop('ZIP.Code', axis=1)

    def handle_missing_values(self):
        """Fills missing numerical values with their respective column medians."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median(numeric_only=True))

    def preprocess(self):
        """
        Runs all preprocessing steps.

        Returns:
        pd.DataFrame: The cleaned and preprocessed dataset.
        """
        self.remove_unnecessary_columns()
        self.handle_missing_values()
        return self.data
