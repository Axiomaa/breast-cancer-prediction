import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

'''
C: Regularization param. Controls trade-off between maximizing the margin and minimizing classification error. High C - aims to classify all points correctly but may lead to overfitting.
Kernel: Testing kernels linear, rbf, poly and sigmoid
Gamma: If we test RBF and polynomial kernels (3D)
'''

class SVMModel:
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
        
    def decision_boundary(self, svc, x_min, x_max, y_min, y_max, resolution=100):
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        zz = svc.predict(grid_points)
        zz = zz.reshape(xx.shape)
        return xx, yy, zz
        
    def test_kernels_and_C(self, train_data, test_data):
        X_train, y_train = train_data.drop('diagnosis', axis=1), train_data['diagnosis']
        X_test, y_test = test_data.drop('diagnosis', axis=1), test_data['diagnosis']
        assert len(X_train) == len(y_train), "X_train and y_train must have the same number of rows."

        # Parameters
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        results = []
        pcs = [0, 1]

        # Reduce to 2D for visualization
        X_train_2d = X_train.iloc[:, pcs]
        
        # Subplots for visualization
        fig, axes = plt.subplots(len(C_values), len(kernels), figsize=(20, 20))
        plot_idx = 0

        # Cross-validation setup
        cross_validation = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        for kernel in kernels:
            for C_value in C_values:

                # Calculating fold-metrics for unbiased results
                for train_idx, val_idx in cross_validation.split(X_train, y_train):
                    svm_classifier = SVC(kernel=kernel, C=C_value, random_state=42)
                    svm_classifier.fit(X_train.iloc[train_idx], y_train.iloc[train_idx]) # Train on training fold
                    y_pred = svm_classifier.predict(X_train.iloc[val_idx]) # Predict on validation fold

                    accuracy = accuracy_score(y_train.iloc[val_idx], y_pred)
                    f1 = f1_score(y_train.iloc[val_idx], y_pred, average='weighted')
                    performance = (accuracy + f1) / 2

                    results.append({
                        'Kernel': kernel,
                        'C': C_value,
                        'F1-Score': f1,
                        'Performance': performance,
                        'Accuracy': accuracy
                    })

                # Separate classifier for the visualizer
                visual_clf = SVC(kernel=kernel, C=C_value, random_state=42)
                visual_clf.fit(X_train_2d, y_train)

                # Decision Boundary Visualization
                ax = axes.flat[plot_idx]
                x_min, x_max = X_train_2d.iloc[:, 0].min(), X_train_2d.iloc[:, 0].max()
                y_min, y_max = X_train_2d.iloc[:, 1].min(), X_train_2d.iloc[:, 1].max()
                
                # Get the decision boundary values and plot the scatterplot
                xx, yy, zz = self.decision_boundary(visual_clf, x_min, x_max, y_min, y_max)
                boundary_cmap = colors.ListedColormap(['lightblue', 'coral'])
                ax.contourf(xx, yy, zz, alpha=0.8, cmap=boundary_cmap)
                scatter = ax.scatter(
                    X_train_2d.iloc[:, 0], 
                    X_train_2d.iloc[:, 1], 
                    c=y_train, 
                    edgecolor='k', 
                    cmap=colors.ListedColormap(['darkblue', 'darkred']), 
                    s=50
                )
                ax.set_title(f"Kernel: {kernel}, C: {C_value}")
                ax.set_xlabel(f"PC1")
                ax.set_ylabel(f"PC2")
                plot_idx += 1

        legend_vis = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.add_artist(legend_vis)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'svm_decision_boundaries.pdf'))
        plt.show()

        results_df = pd.DataFrame(results)
        best_results_df = results_df.loc[results_df.groupby('Kernel')['Performance'].idxmax()]
        print("Best result for each kernel: ")
        print(best_results_df)

        results_file_path = os.path.join(self.output_path, 'svm_kernel_C_results.csv')
        results_df.to_csv(results_file_path, index=False)
        print(f"Results saved to: {results_file_path}")

        # Bar plots for each kernel and c-value with their accuracy score
        palette = sns.color_palette("mako", n_colors=len(results_df['Kernel'].unique()))
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='C', 
            y='Accuracy', 
            hue='Kernel', 
            data=results_df, 
            palette=palette
        )
        plt.title("SVM Accuracy for Different Kernels and C-values")
        plt.xlabel("C-value")
        plt.ylabel("Accuracy")
        plt.legend(title='Kernel', loc='lower right')
        plot_path = os.path.join(self.output_path, 'svm_accuracy_kernels_cvalues.pdf')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()

    ### Add a 3D plot and tune gamma values for non-linear boundaries to test the difference ###
    def plot_3D_svm_params(self, train_data, test_data):
        pass

    def confusion_matrix_final_model(self, train_data, test_data, C=1, kernel='linear'):
        X_train, y_train = train_data.drop('diagnosis', axis=1), train_data['diagnosis']
        X_test, y_test = test_data.drop('diagnosis', axis=1), test_data['diagnosis']

        svm_best = SVC(C=C, kernel=kernel)
        svm_best.fit(X_train, y_train)
        y_pred = svm_best.predict(X_test)
        model_name = "SVM"
        accuracy = accuracy_score(y_test, y_pred)
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        }
        metrics_df = pd.DataFrame([metrics])
        csv_file_path = os.path.join(self.output_path, "model_performance.csv")
        print(f"\nModel accuracy: {accuracy:.4f}\n")
        print(classification_report(y_test, y_pred))
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        else:
            updated_df = metrics_df

        updated_df.to_csv(csv_file_path, index=False)
        print(f"Metrics for {model_name} saved to {csv_file_path}")

        # Confusion matrix plot
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['benign', 'malignant'], cmap='Blues', colorbar=False)
        plt.title("Confusion Matrix for Final SVM Model")
        plt.savefig(os.path.join(self.output_path, 'svm_confusion_matrix_final_model.pdf'))
        plt.show()

        # Save model
        joblib.dump(svm_best, os.path.join(self.model_output_path, 'final_svm_model.pkl'))

    """Note: add a separate plot for the final decision boundary with kernel and C"""

if __name__ == '__main__':
    data_path = "../data"
    output_path = "../output"
    model_output_path = "../models"
    svm_model = SVMModel(data_path, output_path, model_output_path)

    train_data = svm_model.load_data("train_data.csv")
    test_data = svm_model.load_data("test_data.csv")

    if train_data is not None and test_data is not None:
        svm_model.test_kernels_and_C(train_data=train_data, test_data=test_data)

        C_value = float(input("Enter C value to use for SVM: "))
        kernel = str(input("Enter the kernel to use for SVM: "))
        svm_model.confusion_matrix_final_model(train_data=train_data, test_data=test_data, C=C_value, kernel=kernel)
