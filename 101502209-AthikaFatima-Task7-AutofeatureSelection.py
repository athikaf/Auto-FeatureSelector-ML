import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def preprocess_dataset(dataset_path):
    # Load the dataset
    data = pd.read_csv(dataset_path)

    # Sanitize column names to remove special characters
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)

    # Ensure only numeric columns are used for feature selection
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_cols]

    # Handle missing values by filling with mean
    data.fillna(data.mean(), inplace=True)

    # Separate features and target variable
    X = data.drop('Overall', axis=1, errors='ignore')  # Adjust target column if necessary
    y = (data['Overall'] >= 87).astype(int)  # Binary target variable based on 'Overall' >= 87

    # Number of features to select
    num_feats = 30

    return X, y, num_feats



# Pearson Correlation function
def cor_selector(X, y, num_feats):
    cor_list = []
    for col in X.columns:
        cor = np.corrcoef(X[col], y)[0, 1]  # Calculate correlation
        cor_list.append(cor)

    cor_list = [0 if np.isnan(i) else i for i in cor_list]  # Handle NaN values
    cor_abs = np.abs(cor_list)  # Take absolute value

    # Sort features by correlation in descending order
    sorted_indices = np.argsort(cor_abs)[-num_feats:][::-1]
    cor_support = np.zeros(len(cor_list), dtype=bool)
    cor_support[sorted_indices] = True
    cor_feature = X.columns[cor_support]

    return cor_support, cor_feature


# Chi-Square function
def chi_squared_selector(X, y, num_feats):
    # Normalize the feature set to the range [0, 1]
    X_norm = MinMaxScaler().fit_transform(X)

    # Apply Chi-Square test
    chi_selector = SelectKBest(score_func=chi2, k=num_feats)
    chi_selector.fit(X_norm, y)

    # Get the support and selected features
    chi_support = chi_selector.get_support()
    chi_feature = X.columns[chi_support]

    return chi_support, chi_feature


# RFE function
def rfe_selector(X, y, num_feats):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the model (Logistic Regression in this case)
    model = LogisticRegression(max_iter=1000)

    # Apply RFE
    rfe_selector = RFE(estimator=model, n_features_to_select=num_feats, step=1)
    rfe_selector = rfe_selector.fit(X_scaled, y)

    # Get the support and selected features
    rfe_support = rfe_selector.support_
    rfe_feature = X.columns[rfe_support]

    return rfe_support, rfe_feature


# Embedded Logistic Regression function
def embedded_log_reg_selector(X, y, num_feats):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the Logistic Regression model with L1 regularization
    model = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42)

    # Apply SelectFromModel
    embed_selector = SelectFromModel(estimator=model, max_features=num_feats)
    embed_selector.fit(X_scaled, y)

    # Get the support and selected features
    embedded_lr_support = embed_selector.get_support()
    embedded_lr_feature = X.columns[embedded_lr_support]

    return embedded_lr_support, embedded_lr_feature


# Embedded Random Forest function
def embedded_rf_selector(X, y, num_feats):
    # Define the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Apply SelectFromModel
    embed_selector = SelectFromModel(estimator=model, max_features=num_feats)
    embed_selector.fit(X, y)

    # Get the support and selected features
    embedded_rf_support = embed_selector.get_support()
    embedded_rf_feature = X.columns[embedded_rf_support]

    return embedded_rf_support, embedded_rf_feature


# Embedded LightGBM function
def embedded_lgbm_selector(X, y, num_feats):
    # Define the LightGBM model
    model = LGBMClassifier(n_estimators=100, random_state=42)

    # Apply SelectFromModel
    embed_selector = SelectFromModel(estimator=model, max_features=num_feats)
    embed_selector.fit(X, y)

    # Get the support and selected features
    embedded_lgbm_support = embed_selector.get_support()
    embedded_lgbm_feature = X.columns[embedded_lgbm_support]

    return embedded_lgbm_support, embedded_lgbm_feature


def autoFeatureSelector(dataset_path):
    # Preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)

    # Run all methods
    cor_support, _ = cor_selector(X, y, num_feats)
    chi_support, _ = chi_squared_selector(X, y, num_feats)
    rfe_support, _ = rfe_selector(X, y, num_feats)
    embedded_lr_support, _ = embedded_log_reg_selector(X, y, num_feats)
    embedded_rf_support, _ = embedded_rf_selector(X, y, num_feats)
    embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    embedded_lgbm_support = np.array([feature in embedded_lgbm_feature for feature in X.columns])

    # Combine results into a DataFrame
    feature_selection_df = pd.DataFrame({
        'Feature': X.columns,
        'Pearson': cor_support,
        'Chi-2': chi_support,
        'RFE': rfe_support,
        'Logistics': embedded_lr_support,
        'Random Forest': embedded_rf_support,
        'LightGBM': embedded_lgbm_support
    })

    # Count the total selections for each feature
    feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)

    # Sort by Total and return the top features
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    best_features = feature_selection_df.loc[feature_selection_df['Total'] > 0, 'Feature'].tolist()

    return best_features


if __name__ == "__main__":
    dataset_path = input("Enter the path to the dataset: ")

    best_features = autoFeatureSelector(dataset_path=dataset_path)
    print("Best selected features:")
    print(best_features)
