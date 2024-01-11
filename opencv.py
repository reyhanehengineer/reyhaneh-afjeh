import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Dummy dataset (replace this with your actual dataset)
# X: Features, y: Age labels
X = np.random.rand(100, 224, 224, 3)  # Assuming input images of size 224x224x3
y = np.random.randint(1, 100, size=(100,))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but can be beneficial)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# Define and train a simple MLPRegressor model
model = make_pipeline(MLPRegressor(hidden_layer_sizes=(100,), max_iter=500))
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Make predictions on the test set
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')