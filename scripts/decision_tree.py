import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

class DecisionTreeModel:
    def __init__(self, data_path, output_path, model_output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.model_output_path = model_output_path
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(model_output_path, exist_ok=True)

    def load_data(self, filename):
        try:
            filepath = os.path.join(self.data_path, filename)
            data = pd.read_csv(filepath)
            print(f"Datafile '{filename}' loaded successfully.")
            return data
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return None

    def decision_boundary(self, model, train_data, feature_ids=[0, 1], interval=0.05):
        X = train_data.drop(columns='diagnosis')
        y = train_data['diagnosis']

        # Plot bounds
        x_min, x_max = X.iloc[:, feature_ids[0]].min() - 1, X.iloc[:, feature_ids[0]].max() + 1
        y_min, y_max = X.iloc[:, feature_ids[1]].min() - 1, X.iloc[:, feature_ids[1]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, interval), np.arange(y_min, y_max, interval))

        # Preparing feature (PC) grid
        grid = np.c_[xx.ravel(), yy.ravel()]
        feature_grid = pd.DataFrame(np.zeros((grid.shape[0], X.shape[1])), columns=X.columns)
        feature_grid.iloc[:, feature_ids[0]] = grid[:, 0]
        feature_grid.iloc[:, feature_ids[1]] = grid[:, 1]
        
        # Predicting over the grid
        zz = model.predict(feature_grid).reshape(xx.shape)

        # Decision boundary plot
        plt.contourf(xx, yy, zz, alpha=0.3, cmap=colors.ListedColormap(['lightblue', 'lightcoral']))
        scatter = plt.scatter(X.iloc[:, feature_ids[0]], X.iloc[:, feature_ids[1]], c=y, edgecolor='k', cmap='coolwarm', s=50)
        plt.colorbar(scatter)
        plt.xlabel(f'PC{feature_ids[0] + 1}')
        plt.ylabel(f'PC{feature_ids[1] + 1}')
        plt.title("Decision Boundary Plot")
        plt.savefig(os.path.join(self.output_path, 'decision_boundary_plot.pdf'))
        plt.show()

    def cross_validate_model(self, train_data):
        X, y = train_data.drop(columns='diagnosis'), train_data['diagnosis']

        # Storing scores for each depth (3-40) to choose which criterion to use for the final model
        gini_f1_scores, gini_accuracies, entropy_f1_scores, entropy_accuracies = [], [], [], []
        depths = list(range(3,40))

        fig, ax = plt.subplots(figsize=(12, 8))

        criterion_colors = {'gini': 'blue', 'entropy': 'orange'}

        for criterion in ['gini', 'entropy']:
            skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
            avg_f1_values, avg_accuracy_values = [], []
            
            for depth in depths:
                f1_scores, accuracies = [], []

                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
                    accuracies.append(accuracy_score(y_val, y_pred))

                # Calculate avg F1 scores and accuracy for gini and entropy 
                avg_f1_values.append(np.mean(f1_scores))
                avg_accuracy_values.append(np.mean(accuracies))

                if criterion == 'gini':
                    gini_f1_scores.append(np.mean(f1_scores))
                    gini_accuracies.append(np.mean(accuracies))
                elif criterion =='entropy':
                    entropy_f1_scores.append(np.mean(f1_scores))
                    entropy_accuracies.append(np.mean(accuracies))

            ax.plot(depths, avg_f1_values, label=f'{criterion} F1-Score', color=criterion_colors[criterion])
            ax.plot(depths, avg_accuracy_values, label=f'{criterion} Accuracy', linestyle='--', color=criterion_colors[criterion])

        # Cross validation plot
        ax.set_title("Cross validation for 'gini' and 'entropy'")
        ax.set_xlabel("Tree Depth")
        ax.set_ylabel("Score")
        ax.legend()
        plt.savefig(os.path.join(self.output_path, 'decision_tree_cross_validation_results.pdf'))
        plt.show()

        # Determine best scores and depths for gini and entropy to display
        best_gini_accuracy, best_gini_accuracy_depth = max(zip(gini_accuracies, depths))
        best_entropy_accuracy, best_entropy_accuracy_depth = max(zip(entropy_accuracies, depths))
        best_gini_f1, best_gini_f1_depth = max(zip(gini_f1_scores, depths))
        best_entropy_f1, best_entropy_f1_depth = max(zip(entropy_f1_scores, depths))

        best_scores = [
            ['', 'gini', 'entropy'],
            ['accuracy', f'({best_gini_accuracy:.4f}, {best_gini_accuracy_depth})', f'({best_entropy_accuracy:.4f}, {best_entropy_accuracy_depth})'],
            ['f1-score', f'({best_gini_f1:.4f}, {best_gini_f1_depth})', f'({best_entropy_f1:.4f}, {best_entropy_f1_depth})']
        ]

        df_best_scores = pd.DataFrame(best_scores)
        print(f'\nBest F1-scores and accuracy for gini and entropy and at what depth: ')
        print(f'{df_best_scores}\n')

    def train_final_model(self, train_data, test_data, criterion='gini', max_depth=8):
        X_train, y_train = train_data.drop(columns='diagnosis'), train_data['diagnosis']
        X_test, y_test = test_data.drop(columns='diagnosis'), test_data['diagnosis']
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Classification report
        y_pred = model.predict(X_test)
        model_name = "Decision Tree"
        accuracy = accuracy_score(y_test, y_pred)
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        }
        metrics_df = pd.DataFrame([metrics])
        csv_file_path = os.path.join(output_path, "model_performance.csv")
        print(f"\nModel accuracy: {accuracy:.4f}\n")
        print(classification_report(y_test, y_pred))
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        else:
            updated_df = metrics_df

        updated_df.to_csv(csv_file_path, index=False)
        print(f"Metrics for {model_name} saved to {csv_file_path}")

        # Plotting the confusion matrix to evaluate model performance
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['benign', 'malignant'], cmap='Blues')
        plt.title("Confusion Matrix for Final Model")
        plt.savefig(os.path.join(self.output_path, 'decision_tree_confusion_matrix_final_model.pdf'))
        plt.show()
        
        # Decision tree plot to visualize nodes based on PCs
        plt.figure(figsize=(10, 8))
        plot_tree(
            model, 
            feature_names=X_train.columns.tolist(),
            class_names=['benign', 'malignant'],
            filled=True,
            rounded=True,
            precision=4
        )
        plt.title('Decision Tree Structure for Breast Cancer Classification')
        plt.xlabel('Features and Decision Thresholds')
        plt.ylabel('Tree Depth')
        plt.savefig(os.path.join(self.output_path, 'decision_tree_plot.pdf'))
        plt.show()
        
        # Save model
        joblib.dump(model, os.path.join(self.model_output_path, 'final_decision_tree_model.pkl'))

        return model

# Running the model
if __name__ == "__main__":
    data_path = "../data"
    output_path = "../output"
    model_output_path = "../models"
    dt_model = DecisionTreeModel(data_path, output_path, model_output_path)

    train_data = dt_model.load_data("train_data.csv")
    test_data = dt_model.load_data("test_data.csv")

    if train_data is not None and test_data is not None:
        dt_model.cross_validate_model(train_data)

        chosen_criterion = input("Enter criterion ('gini' or 'entropy'): ")
        chosen_depth = int(input("Enter the depth to use for the final Decision Tree model: "))
        model = dt_model.train_final_model(train_data, test_data, criterion=chosen_criterion, max_depth=chosen_depth)

        dt_model.decision_boundary(model, train_data, feature_ids=[0,1], interval=0.05)
