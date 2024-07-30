from src.data.load_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance
from src.feature.build_features import create_features
from src.model.train_model import train_model
from src.model.predict_model import predict_model


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/real_estate.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_features(df)

    # Train linear regression model
    model, X_train, y_train, X_test, y_test,train_mae,train_r2 = train_model(X, y)
    print(f"Train MAE: {train_mae}")
    print(f"Train R2: {train_r2}")
    # Predict & Evaluate the model
    R2, test_mae = predict_model(model, X_test, y_test)
    print(f"Test R-squared: {R2}")
    print(f"Test MAE:{test_mae}")