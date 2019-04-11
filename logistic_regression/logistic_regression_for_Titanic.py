# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

style.use('ggplot')

#READING IN THE TITANIC DATASET CSV
train = pd.read_csv('')

#PEEKING AT THE DATASET
train.head()

'''
MISSING DATA
'''


#SOMETIMES YOU WILL HAVE MISSING DATA IN YOUR DATASET, WE CAN PLOT A HEATMAP TO SEE WHERE AND HOW MANY MISSING DATA WE HAVE

nan = train.isnull()

sns.heatmap(nan, yticklabels = False, cbar = False, cmap = 'viridis') # every yellow on graph shows missing data!

'''
DATA ANALYSIS
'''

sns.set_style('whitegrid')

# NUMBER OF PEOPLE WHO SURVIVED VS PEOPLE WHO DIDN'T SURVIVED
sns.countplot(x = 'Survived', data = train)

# LESS PEOPLE SURVIVED BUT PEOPLE WHO SURVIVED ARE TWICE LIKELY TO BE FEMALE
sns.countplot(x = 'Survived', hue = 'Sex', data = train)

# PEOPLE WHO DID NOT SURVIVED ARE MOSTLY 3RD CLASS(CHEAPEST) AND PEOPLE WHO SURVIVED ARE LEANING MORE TOWARDS 1ST CLASS
sns.countplot(x='Survived',hue='Pclass',data=train)

# LOOKS LIKE A BIMODAL DISTRIBUTION
sns.distplot(train['Age'].dropna(),kde=False,bins=30)

# MOST PEOPLE ONBOARD DIDN'T HAVE EITHER A CHILDREN OR SPOUSE ONBOARD GROUP 1 ARE MOST LIKELY SPOUSE
sns.countplot(x='SibSp',data=train)

#PLOTTING A HISTOGRAM
train['Fare'].hist(bins=40,figsize=(10,4))


# DATA CLEANING

plt.figure(figsize = (12, 7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train) # wealthier passenger tend to be older

#HANDLING MISSING DATA VALUES
def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]
	
	if pd.isnull(Age):
		if Pclass == 1:
			return 37
		elif Pclass == 2:
			return 29
		else:
			return 24
	else:
		return Age

#APPLYING THE FUNCTION ON 'AGE' COLUMN TO RETURN A NON NULL AGE COLUMN
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)

#PLOTTING A HEATMAP FOR NULL VALUES
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

#DROP THE CABIN COLUMN AS THERE ARE TOO MANY MISSING DATA
train.drop('Cabin',axis=1,inplace = True)

#PLOTTING A HEATMAP FOR NULL VALUES(TO SEE IF THERE IS STILL ANY NULL VALUES)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')

# SEEING AS THERE ARE ONLY ONE MISSING VALUE LEFT, WE CAN DROP THE CORRESPONDING ROW
train.dropna(inplace=True)

# NO MORE MISSING DATA, WE ARE READY!
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')

'''
CONVERTING CATEGORICAL FEATURES

WE’LL NEED TO CONVERT CATEGORICAL FEATURES TO DUMMY VARIABLES USING PANDAS! OTHERWISE OUR MACHINE LEARNING ALGORITHM WON’T BE ABLE TO DIRECTLY TAKE IN THOSE FEATURES AS INPUTS.
'''

sex =pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)

train = pd.concat([train, sex, embark], axis = 1)

#DROP COLUMNS OF DATA THAT ARE NOT USEFUL OR NOT GOOD FOR ML ALGO
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis = 1, inplace = True)

train.head()

'''
SPLITTING DATA SETS
'''

train.columns

#CONVERTING THE DATAFRAME OF FEATURES TO A NUMPY ARRAY AND DROPPING THE LABEL COLUMN
X = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]

#CONVERTING THE DATAFRAME OF 'Survived' TO A NUMPY ARRAY
y = train['Survived']

# SPLITTING THE DATA INTO 80% - 20% RATIO. 80% FOR TRAINING X & Y AND 20% FOR TESTING X & Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#INVOKING THE REGRESSION MODEL
logmodel = LogisticRegression()

#FITTING THE BEST FIT LINE INTO THE TRAINING DATASET
logmodel.fit(X_train, y_train)

#MAKING PREDICTIONS
predictions = logmodel.predict(X_test)

# PRINTING THE EVALUATION OF OUR MODEL
print(classification_report(y_test,predictions))

#OUTPUTING THE CONFUSION MATRIX EVALUATION
confusion_matrix(y_test,predictions)