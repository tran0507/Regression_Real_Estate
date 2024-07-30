# Import accuracy score
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

# # Function to predict and evaluate
def predict_model(model, X_test, y_test):
    # Predict based on the testing set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    R2 = r2_score(y_test, y_pred)
      # we need mean absolute error   
    test_mae = mean_absolute_error(y_test,y_pred)

    return {R2,test_mae }
