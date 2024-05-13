import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Create a directory for plots
if not os.path.exists('pca_plots'):
    os.makedirs('pca_plots')

# Read in the data, clean the variable names
healthData = pd.read_csv("dataset/Health-Sciences-Data-File-New-(project).csv")
healthData = healthData.iloc[:, :35]
healthData.columns = healthData.columns.str.replace(' ', '_').str.lower()
healthData['sex'] = healthData['sex'].map({'M': 0, 'F': 1})

# For each variable/column, count the NAs, if there's at least one, print the column name and NA count
for i in healthData.columns:
    NAs = healthData[i].isna().sum()
    if NAs > 0:
        print(i, NAs)

# Remove all the columns that had NA counts higher than 5% of the total observations (~314)
healthData = healthData.drop(columns=['bia_percent_fat', 'sf_1', 'sf_2', 'sf_3', 'waist', 'pl_3', 'hr_3', 'rpe_3', 'date', 'idnum'])

# Apply PCA without specifying the number of components
pca = PCA()
pca.fit(healthData)

# Plot the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o')
plt.title('Explained Variance by Different Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('pca_plots/explained_variance.png')
plt.close()

# Calculate the correlation matrix
corr_matrix = pd.DataFrame(pca.components_, columns=healthData.columns).T.corr()

# Plot the heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Principal Components and Original Features')
plt.savefig('pca_plots/correlation_heatmap.png')
plt.close()

# PCA with number of components based on plot
n_components = 9 # Update this based on number of components
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(healthData)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC' + str(i) for i in range(1, n_components + 1)])

# Create scatterplot matrices for all the principal components vs ff_total
for i in range(1, n_components + 1):
    plt.figure()
    plt.scatter(principalDf['PC' + str(i)], healthData['fftotal'])
    plt.xlabel('PC' + str(i))
    plt.ylabel('fftotal')
    plt.savefig(f'pca_plots/PC{i}_vs_fftotal.png')
    plt.close()

# Create correlation matrices for all the principal components and ff_total
# Concatenate the principal components and the target variable
data_with_pca = pd.concat([principalDf, healthData['fftotal']], axis=1)

# Calculate the correlation matrix
corr_matrix = data_with_pca.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Principal Components and fftotal')
plt.savefig('pca_plots/correlation_heatmap_pca.png')
plt.close()

# Define target variable
target = 'fftotal'

# Randomly split principalDf and target into training and testing datasets
trainingdata, testdata, trainingvalues, testvalues = train_test_split(principalDf, healthData[target], test_size=0.5, random_state=42)

# Build the models using the training data
model1 = LinearRegression().fit(trainingdata, trainingvalues)
model2 = LinearRegression().fit(trainingdata.drop(columns=['PC' + str(n_components)]), trainingvalues)



print(model1)
print(model2)


# Print the details of model1
print("Details of model1:")
print(dir(model1))

# Print the details of model2
print("\nDetails of model2:")
print(dir(model2))


# Print the coefficients of model1
print("Coefficients of model1:")
print(model1.coef_)

# Print the intercept of model1
print("Intercept of model1:")
print(model1.intercept_)

# Print the coefficients of model2
print("Coefficients of model2:")
print(model2.coef_)

# Print the intercept of model2
print("Intercept of model2:")
print(model2.intercept_)


# Create a table of the models and their adjusted R^2 values
modelRSq = pd.DataFrame({
    'modelNum': [1, 2],
    'RSq': [
        model1.score(trainingdata, trainingvalues),
        model2.score(trainingdata.drop(columns=['PC' + str(n_components)]), trainingvalues)
    ]
})

print(modelRSq)

# Using each model, predict ff_total for the testing data set
testvalues = testvalues.to_frame()
testvalues['pred'] = model1.predict(testdata)
testvalues['pred2'] = model2.predict(testdata.drop(columns=['PC' + str(n_components)]))

# Create a function to calculate the RMSE
def RMSE(actual, prediction):
    return np.sqrt(mean_squared_error(actual, prediction))

# Find the RMSE of all the models
modelRMSE = pd.DataFrame({
    'model': [1, 2],
    'RMSE': [
        RMSE(testvalues['fftotal'], testvalues['pred']),
        RMSE(testvalues['fftotal'], testvalues['pred2'])
    ]
})
print(modelRMSE)
