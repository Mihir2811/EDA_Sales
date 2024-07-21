import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('SampleSuperstore.csv')
print('------------------------------------------------------------------------------------------------------------------------------------------')

data

print('------------------------------------------------------------------------------------------------------------------------------------------')


#cleaning the data and do the necessary preprocessing

data.dropna(inplace=True)

# Converting categorical features to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category'])

# Normalize numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_features = ['Sales', 'Quantity', 'Discount', 'Profit']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print('------------------------------------------------------------------------------------------------------------------------------------------')


# prompt: Using dataframe data: create a detailed EDA of the data

# Assuming 'data' is a pandas DataFrame, the following code performs EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Basic information about the DataFrame
print(data.info())

print('------------------------------------------------------------------------------------------------------------------------------------------')


# Summary statistics for numerical columns
print(data.describe())

print('------------------------------------------------------------------------------------------------------------------------------------------')


# Check for missing values
print(data.isnull().sum())

print('------------------------------------------------------------------------------------------------------------------------------------------')


# Visualize the distribution of numerical columns
for col in data.select_dtypes(include=['number']):
    plt.figure()
    sns.histplot(data[col])
    plt.title(f'Distribution of {col}')
    plt.show()
    print('------------------------------------------------------------------------------------------------------------------------------------------')


# Explore categorical columns
for col in data.select_dtypes(include=['object', 'category']):
    plt.figure()
    sns.countplot(data[col])
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.show()
    print('------------------------------------------------------------------------------------------------------------------------------------------')


# Pairwise relationships between numerical columns
sns.pairplot(data.select_dtypes(include=['number']))
plt.show()


# Correlation matrix for numerical columns
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()
