from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score


# Function to train the model
def train_model(X, y):
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # train your model
    model = LinearRegression().fit(X_train,y_train)

    # make preditions on train set
    train_pred = model.predict(X_train)
     #evaluate your model
    # we need mean absolute error   
    train_mae = mean_absolute_error(y_train,train_pred)
    train_r2=r2_score(y_train,train_pred)


    # Save the trained model
    with open('model/linearregression.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, X_train, y_train, X_test, y_test, train_mae, train_r2
