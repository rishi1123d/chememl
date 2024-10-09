
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv(file_path)
    data_cleaned = data.dropna(subset=['activity'])
    data_cleaned['pubchem_smiles_length'] = data_cleaned['pubchem_smiles_cleaned'].str.len()
    data_cleaned['alogps_smiles_length'] = data_cleaned['alogps_smiles_cleaned'].str.len()
    return data_cleaned

def prepare_features_and_target(data):
    """Prepare features and target variables."""
    X = data[['pubchem_smiles_length', 'alogps_smiles_length']]
    y = data['activity']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train a model and evaluate its performance."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('all_smiles_data.csv')
    
    # Prepare features and target
    X_train, X_test, y_train, y_test = prepare_features_and_target(data)
    
    # Define models
    models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Neural Network": MLPRegressor(max_iter=500)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        mse = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = mse
    
    # Print results
    for name, mse in results.items():
        print(f"{name} MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
