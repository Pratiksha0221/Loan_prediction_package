import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Visualizer:
    """
    A class for visualizing various aspects of loan prediction data.
    
    Attributes:
    data (pd.DataFrame): The dataset to visualize.
    """
    
    def __init__(self, data):
        """
        Initializes the Visualizer with the dataset.
        
        Parameters:
        data (pd.DataFrame): The dataset to visualize.
        
        Example:
        >>> visualizer = Visualizer(data)
        >>> visualizer.visualize_data()
        """
        self.data = data
    
    def visualize_data(self):
        """
        Generates plots to explore the dataset.
        
        Creates histograms of the features and checks for the 'Personal.Loan' column.
        
        Example:
        >>> visualizer.visualize_data()
        """
        if 'Personal.Loan' not in self.data.columns:
            print("Column 'Personal.Loan' not found. Skipping loan-related plots.")
            return
        
        # Histogram Representation of features
        self.data.hist(figsize=(20, 15), color='blue')
        plt.show()
