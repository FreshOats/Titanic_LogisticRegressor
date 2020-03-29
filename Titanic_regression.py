## Titanic - A Logistic Regression Predicitve Analysis
import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
#print(passengers.head())  # Just to check the headings and see data

# Update sex column to numerical - this converts the classifiers of male and female to binaries, 0 and 1 respectively. I opted to use a lambda function rather than employing a for loop or .map()
passengers['Sex'] = passengers['Sex'].apply(lambda x: 0 if x=='male' else 1)
print(passengers.head())  # Just observing that the outcome actually changed the data

# Fill the nan values in the age column -- fill with the average age of passengers
passengers['Age'] = passengers.Age.fillna(passengers.Age.mean())
print(passengers['Age'])  # From the above table, it's clear that there were nan values in the age, so used .fillna() to fill those values with the mean age of the passengers

# Create a first class column - this will be a new column to identify first class passengers as 1 and all other passengers as 0
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x==1 else 0)
#print(passengers.head())  # Verify 

# Create a second class column to represent binaries, 1 for second class, 0 for all others. 
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x==2 else 0)
#print(passengers['SecondClass'])  # Verify


# We want to compare the Sex, Age, and Class of the passengers with the Survival - so split these into two different data frames. Note that to keep the passengers as a df with the list of columns, must be entered as [[ ..., ..., ..., ...]]
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']
#print(features)
#print(survival)



# Perform train, test, split using the format:
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state= 6)
features_train, features_test, survival_train, survival_test = train_test_split(features, survival, train_size = .8, test_size = 0.2, random_state = 5)

#Look at the shapes of the training data sets using the format: 
# print([x_train.shape, x_test.shape, y_train.shape, y_test.shape])
print([features_train.shape, features_test.shape, survival_train.shape, survival_test.shape])


# Since sklearn uses Regularization in its implementation, the data needs to be normalized first. StandardScaler() will do this to the data.  Note: the Survival set is already binary and shouldn't be scaled
# Scale the feature data so it has mean = 0 and standard deviation = 1

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Next we establish the Logisitic Regression model using LogisticRegression()
regressor = LogisticRegression()
regressor.fit(features_train, survival_train)

# After creating the model, we need to calculate the score for both the training set and the test set. 
score_train = regressor.score(features_train, survival_train) # Training Model Score
score_test = regressor.score(features_test, survival_test)  # Test Model Score
print([score_train, score_test])  # The scores yielded were [0.7935393258426966, 0.8156424581005587]



# Determine the coefficients for the importance of the model
coefficients = regressor.coef_
coefficients = coefficients.tolist()[0]
print(coefficients) # We want to observe the most important factors - the highest magnitude coefficient is the strongest predictor of survival. Reported in the order: Sex, Age, 1st Class, 2nd Class 
#[1.2471930445189796, -0.41823212413550115, 0.9701532325930738, 0.502171961943298]
# According to these data, the most important factor was gender followed by 1st class. Even second class took a higher coefficient than age, which suggests that the lower, possibly 3rd class had a lot of children that died or that they just simply didn't survive the temperatures despite being put on the boats. 

# Testing some sample passengers
# Sample passenger features
# Remember [['Sex', 'Age', 'FirstClass', 'SecondClass']]
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Joobs = np.array([0.0,39.0,0.0,1.0])

# Combine passenger arrays into a list - this MUST be done before making predictions
sample_passengers = [Jack, Rose, Joobs]

# Since the training data and test data were scaled, the sample_passenger data must also be scaled using .transform()
sample_passengers = scaler.transform(sample_passengers)


# Finally, we can predict who will survive the Titanic sinking
prediction = regressor.predict(sample_passengers)
probabilities = regressor.predict_proba(sample_passengers)
print(prediction)  # This predicts that Jack and Joobs die. Rose lives.
print(probabilities)  # Represented as P(dying), P(surviving)
'''
[[0.88699391 0.11300609] Jack
 [0.05330355 0.94669645] Rose
 [0.80149366 0.19850634]] Joobs
 '''
# So Rose had a very high probability of survival. Jack in his 3rd class seat only had an 11% chance of survival, and Joobs had a slightly higher 19% change of survival... but he died anyway. 