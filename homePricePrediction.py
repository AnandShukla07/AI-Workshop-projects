from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

housing_data = pd.read_csv('Housing.csv')
# Step 1: Preprocessing
# Convert categorical variables to numerical values using Label Encoding
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea', 'furnishingstatus']

# Apply Label Encoding
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    housing_data[column] = le.fit_transform(housing_data[column])
    label_encoders[column] = le

# Step 2: Define features and target variable
X = housing_data.drop('price', axis=1)  # Features
y = housing_data['price']  # Target

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2

plt.figure(figsize=(8, 5))
sns.histplot(housing_data['price'], kde=True, bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Boxplot for area
plt.figure(figsize=(8, 5))
sns.boxplot(x=housing_data['area'])
plt.title('Boxplot of House Areas')
plt.xlabel('Area')
plt.show()

# Scatter plot: Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=housing_data['area'], y=housing_data['price'])
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# Pairplot for selected features
selected_features = ['area', 'price', 'bedrooms', 'bathrooms']
sns.pairplot(housing_data[selected_features], diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = housing_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

