import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("weatherHistory.csv")

print(data.head())

#check for missing values
#print(data.isnull().sum())

# if there are missing values 
# data = dat.fillna(data.mean())

#print(data.head())

#covert Date column to datetime
data['Date'] = pd.to_datetime(data['Formatted Date'], utc=True)

# Data Vis

encoder = LabelEncoder()
encoded_precip = encoder.fit_transform(data[['Precip Type']])

# We only want numeric data for heatmaps
data_numeric = data.select_dtypes(include=['number'])

plt.figure(figsize=(10, 8))
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm')
#plt.show()

#We want to predict temperature so that will be our target. 
#We are going to split our data in 2 categories testing and training.
#the testing data typically 20% of the data whereas training is 80%



X = data_numeric[['Precip Type']]
Y = data['Temperature (C)']

# Remove rows with NaN values from both X and Y
X_clean = X.dropna()
Y_clean = Y[X_clean.index]


X_train, X_test, y_train, y_test = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)


# Checking the shapes of the splits
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

#Build the linear regression model
model = LinearRegression()

#train the model on the training data
model.fit(X_train, y_train)

#Make predictions on the test data
y_pred = model.predict(X_test)

#Evaluating the model using MSE and R squared
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Example: Predict the temperature for a new day
new_day_data = [[0.59, 12, 260, 12, 0, 1015]]
predicted_temp = model.predict(new_day_data)
print(f"Predicted Temperature: {predicted_temp[0]}°C")


