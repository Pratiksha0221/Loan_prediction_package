from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """
    A class for training a Random Forest model for loan prediction.
    """
    
    def __init__(self, data):
        """
        Initializes the ModelTrainer with the dataset.
        
        Parameters:
        data (pd.DataFrame): The preprocessed dataset containing features and target variable.
        """
        self.data = data
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, test_size=0.25, random_state=0):
        """
        Splits the dataset into training and testing sets.
        
        Parameters:
        test_size (float): Proportion of the dataset for the test set.
        random_state (int): Seed for reproducibility.
        """
        X = self.data.drop(columns=['Personal.Loan'])
        y = self.data['Personal.Loan']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def train_model(self, n_estimators=10, random_state=0):
        """
        Trains a Random Forest model on the dataset.
        
        Parameters:
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluates the trained model using AUC score and Mean Squared Error.
        
        Returns:
        dict: A dictionary with AUC score and MSE.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
        predictions = self.model.predict(self.x_test)
        auc_score = roc_auc_score(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)
        
        # Convert predictions to binary if they are continuous (for classification)
        predictions_binary = (predictions > 0.5).astype(int)
        
        print(f"AUC Score: {auc_score}")
        print(f"Mean Squared Error: {mse}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(self.y_test, predictions_binary)
        return {"AUC Score": auc_score, "Mean Squared Error": mse}

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plots the confusion matrix for model evaluation.
        
        Parameters:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted values by the model.
        """
        conf_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Not Approved', 'Approved'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Random Forest")
        plt.show()
