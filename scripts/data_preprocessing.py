import pandas as pd
import os
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.exceptions  import NotFittedError # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

def load_data(filename, data_path):
    try:
        filepath = os.path.join(data_path, filename)
        data = pd.read_csv(filepath)
        print(f'Datafile loaded successfully\n')
        return data
    except FileNotFoundError:
        print(f'Error: The file "{filename}" was not found.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

def preprocess_data(data):
    try:
        # Drop 'Unnamed: 32' if it exists in the df
        if data.iloc[:, -1].isna().all():
            last_col_name = data.columns[-1]
            data.drop(columns=[last_col_name], inplace=True)
            print(f"Dropped empty column: '{last_col_name}'")
        else:
            print("No completely empty column found, proceeding to dropping missing values.")

        # Drop rows with missing values
        initial_count = data.shape[0]
        data.dropna(inplace=True)
        final_count = data.shape[0]
        dropped_count = initial_count - final_count
        print(f'Rows dropped due to missing values: {dropped_count}')

        if data.empty:
            print("No data left after dropping rows with missing values. Cannot proceed with scaling.")
            return data
        
        # Encode categorical labels "M" and "B"
        try:
            label_encoder = LabelEncoder()
            data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])
            data.drop(['id'], axis=1, inplace=True)
            print(f'\nLabel encoding applied successfully.')
            print("Data after Label Encoding:")
            print(data.head())
        except Exception as e:
            print(f'Error during label encoding: {e}')
            return

        return  data

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
    
    return None, None, None

def eda_feature_exploration(data):
    print(f"\nSummary statistics: ")
    print(data.describe())

    # Feature exploration
    feature_df = pd.DataFrame(columns=['variable_name', 'dtype', 'missing_percentage', 'flag', 'unique_values'])

    for col in data.columns:
        variable_name = col
        dtype = data[col].dtype
        missing_percentages = data[col].isnull().mean() * 100
        unique_values = data[col].nunique()
        flag = 'categorical' if dtype == 'object' else 'numeric'

        feature_df = pd.concat([feature_df, pd.DataFrame({
            'variable_name': [col], 
            'dtype': [dtype], 
            'missing_percentage': [missing_percentages], 
            'flag': [flag], 
            'unique_values': [unique_values]
        })], ignore_index=True)

    return feature_df

def correlation(df):
    variables = [col for col in df.columns if col != 'diagnosis']
    vars_to_remove = []
    print(f"\nRedundant variable pairs: ")

    # Identify pairs of highly correlated features (correlation > 0.9)
    for i, j in zip(*np.where(np.abs(np.triu(df[variables].corr())) > 0.9)):
        if i != j:
            print(f'{variables[i]}, {variables[j]} with corr. {df.corr().iloc[i,j].round(4)}')
            # Ensure only one feature in each pair is added
            if variables[i] not in vars_to_remove and variables[j] not in vars_to_remove:
                vars_to_remove.append(variables[i])

    data_cleaned = df.drop(vars_to_remove, axis=1)

    return data_cleaned

def heatmap_plot(df):

    plt.figure(figsize=(15,10))
    sns.heatmap(
        df.corr(), 
        vmin=-1, 
        vmax=1, 
        cmap='coolwarm', 
        annot=True,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plotname = "heatmap_correlation.pdf"
    try:
        if not os.path.exists(output_folder):
            raise FileNotFoundError(f"The directory {output_folder} does not exist.")
        save_path = os.path.join(output_folder, plotname)
        plt.title(f'Correlation Heatmap of Features', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(save_path)
        plt.tight_layout()
        plt.show()
        print(f"\nHeatmap plot saved as {plotname}")
    except Exception as e:
        print(f"Error: Could not save the heatmap plot. Details: {e}")


def plot_feature_distribution(data, target_column='diagnosis', labels=('Malignant', 'Benign')):
    try:
        # Creates figure and axes for subplots excluding diagnosis
        num_features = len(data.columns) - 1
        ncols = 5
        nrows = (num_features // ncols) + (num_features % ncols > 0)
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
        ## like figsize 20, nrows *3

        for col, ax in zip(data.columns.drop(target_column), axes.flatten()):
            try:
                sns.kdeplot(data=data[data[target_column] == 1], x=col, color='red', label=labels[0], ax=ax, fill=True)
                sns.kdeplot(data=data[data[target_column] == 0], x=col, color='blue', label=labels[1], ax=ax, fill=True)
            except Exception as e:
                print(f"Error plotting {col}: {e}")

        for ax in axes.flatten()[num_features:]:
            ax.remove()

        ## Max (plus padding)
        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=4)

        #fig.subplots_adjust(top=0.85, hspace=1.5)

        handles, _ = ax.get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, ncol=2, title="Diagnosis", loc="upper center", bbox_to_anchor=(0.5, 0.99))
        #plt.suptitle("Feature Distributions by Diagnosis", fontsize=16, weight='bold', y=1.1)

        plotname = "feature_distribution.pdf"
        try:
            if not os.path.exists(output_folder):
                raise FileNotFoundError(f"The directory {output_folder} does not exist.")
            save_path = os.path.join(output_folder, plotname)
            #fig.savefig(save_path)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"\nDistribution plot saved as {plotname}")
        except Exception as e:
            print(f"Error: Could not save the distribution plot. Details: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        plt.show()

## Issues here still
def standard_scale(data, prints=False):

    # Split features and target
    features = data.drop('diagnosis', axis=1)
    target = data['diagnosis']
    print(f'\nFeatures (X) data: ')
    print(features.head())
    print(f'\nTarget (y) data: ')
    print(target.head())

    # Standardize feature values to help normalize distribution
    try: 
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        if prints:
            print(f"\nStandard scaling applied successfully.")
            print(f"Scaled features: \n{features_scaled[:5]}")

        data_scaled = features.copy()
        data_scaled[:] = features_scaled
        data_scaled['diagnosis'] = target

        return data_scaled, features_scaled, target
        
    except ValueError as ve:
        print(f"Error during scaling: {ve}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during scaling: {e}")
        return None, None, None

def prinicpal_component_analysis(feature_df):
    try:
        pca = PCA()
        try:
            pca.fit(feature_df)
        except Exception as e:
            raise RuntimeError(f"An error occurred during PCA fitting: {e}")
        
        explained_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)

        # y-axis range
        y_min, y_max = explained_var.min(), explained_var.max()
        y_min = np.floor(y_min / 10) * 10
        y_step = (y_max - y_min) / 10 if (y_max - y_min) >= 10 else 5
        y_ticks = np.arange(y_min, y_max + y_step, y_step)

        plt.figure(figsize=(10, 8))
        plt.axhline(100, linestyle='--', color='gray')
        plt.plot(explained_var, color='#4A90E2', label='Explained Variance')

        points = plt.scatter(range(len(explained_var)), explained_var, color='blue', label='Data Points')

        for i, txt in enumerate(explained_var):
            plt.annotate(f"{txt:.1f}", (i, txt), textcoords="offset points", xytext=(0, 5), ha='center')

        plt.title("PCA Explained Variance by Number of Components", fontsize=16, weight='bold')
        plt.xlabel('Number of Principal Components')
        plt.xticks(np.arange(0, len(explained_var), 5))
        plt.yticks(y_ticks)
        plt.ylabel('Explained Variance (%)')
        plt.legend()

        filename_pca = 'pca_explained_variance.pdf'
        save_path = os.path.join(output_folder, filename_pca)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"PCA plot saved successfully as {filename_pca}")

        plt.show()

        num_components = int(input("Enter the number of components to retain based on the plot: "))
        return num_components

    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def transform_with_pca(feature_df, target_df, num_components):
    try: 
        pca = PCA(n_components=num_components)

        feature_pca = pca.fit_transform(feature_df)

        pca_df = pd.DataFrame(feature_pca, columns=[f'PC {i+1}' for i in range(num_components)])
        pca_df['diagnosis'] = target_df.values

        print(f"\nPCA transformation complete. DataFrame head: ")
        print(pca_df.head())

        return pca_df
    
    except Exception as e:
        print(f"Error during PCA transformation: {e}")

def visualize_pca_2d(feature_df, target_df):
    try: 

        if isinstance(feature_df, np.ndarray):
            feature_df = pd.DataFrame(feature_df, columns=[f"Feature {i+1}" for i in range(feature_df.shape[1])])

        pca = PCA(n_components=2)
        feature_pca = pca.fit_transform(feature_df)

        feature_names = feature_df.columns

        plt.figure(figsize=(10,8))
        plt.scatter(feature_pca[target_df == 1, 0], feature_pca[target_df == 1, 1], color='coral', label='Malignant')
        plt.scatter(feature_pca[target_df == 0, 0], feature_pca[target_df == 0, 1], color='blue', label='Benign')

        scaling_factor = 10
        for i in range(2):
            plt.arrow(0, 0, pca.components_[0, i] * scaling_factor, pca.components_[1, i] * scaling_factor, width=0.08, color='darkred')
            plt.annotate(f'PC{i+1}', (pca.components_[0, i] * scaling_factor + 0.1, pca.components_[1, i] * scaling_factor), fontsize=12, color='black', fontweight='bold')

        plt.title('Visualization of Prinicipal Components 1 & 2')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()

        filename_pca_2d = 'pca_2d.pdf'
        save_path = os.path.join(output_folder, filename_pca_2d)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nPCA plot saved successfully as {filename_pca_2d}")

        plt.show()

    except Exception as e:
        print(f"Error during PCA 2D visualization: {e}")

def class_balance_plot(target_df):
    class_count = target_df.value_counts()

    colors = ['lightblue' if label == 0 else 'lightcoral' for label in class_count.index]
    plt.bar(class_count.index, class_count.values, color=colors, label='Unbalanced Target Class')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.title('Malignant vs. Benign count')
    plt.xticks([0, 1], ['Benign', 'Malignant'])

    plotname = 'diagnosis_count_barplot.pdf'
    save_path = os.path.join(output_folder, plotname)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nDiagnosis count bar plot saved successfully as {plotname}")

    plt.show()

def prepare_model_data(pca_df):
    try:
        x = pca_df.drop(columns=['diagnosis'])
        y = pca_df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        print(f"\nTraining input shape: {X_train.shape}, Training output shape: {y_train.shape}")
        print(f"\nTesting input shape: {X_test.shape}, Testing output shape: {y_test.shape}")

        test_data = X_test.copy()
        test_data['diagnosis'] = y_test

        train_data = X_train.copy()
        train_data['diagnosis'] = y_train

        test_save_path = os.path.join(data_path, 'test_data.csv')
        train_save_path = os.path.join(data_path, 'train_data.csv')
        test_data.to_csv(test_save_path, index=False)
        train_data.to_csv(train_save_path, index=False)

    except Exception as e:
        print(f"Error preparing model data: {e}")


if __name__ == '__main__':

    # Set output folder relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    global data_path
    data_path = os.path.join(script_dir, '../data')
    os.makedirs(data_path, exist_ok=True)
    print(f"Data output folder set to: {data_path}")

    global output_folder
    output_folder = os.path.join(script_dir, '../output')
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder set to: {output_folder}")

    # Read saved Breast Cancer Wisconsin (Diagnostic) Data Set
    data = load_data('data.csv', data_path)

    print("Data Summary:")
    print(data.describe())

    # Preprocess data
    data_cleaned = preprocess_data(data)

    # Features exploration of the data
    feature_eda_df = eda_feature_exploration(data_cleaned)
    print(f"\nFeature exploration results:")
    print(feature_eda_df)

    # Correlation analysis
    correlation_clean_df = correlation(data_cleaned)

    # Heatmap plot
    heatmap_plot(correlation_clean_df)

    # Feature distribution plot
    plot_feature_distribution(correlation_clean_df, 'diagnosis')

    # Scale data
    data_scaled, features_scaled, target = standard_scale(correlation_clean_df, prints=False)

    # PCA
    num_components = prinicpal_component_analysis(features_scaled)
    pca_df = transform_with_pca(features_scaled, target, num_components)

    # Visualize PC1 and PC2
    visualize_pca_2d(features_scaled, target)

    # Class Balance plot
    class_balance_plot(target)

    # Prepare data for model building
    prepare_model_data(pca_df)

    