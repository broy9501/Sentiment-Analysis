import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess_text

# Read the dataset
filePath = 'merged_sentiment_analysis.csv'
data = pd.read_csv(filePath)

# Preprocess the dataset
data['text'] = data['text'].apply(preprocess_text)


# Set the training split
'''
train_val = 80%
test = 10%
val = 10%

val = 1/9 * 0.8 = 0.10
'''
# train, test = train_test_split(data, test_size=0.2, random_state=42)
train_val, test = train_test_split(data, test_size=0.1, random_state=42, stratify=data['sentiment'])
train, val = train_test_split(train_val, test_size=1/9, random_state=42, stratify=train_val['sentiment'])

'''
Set the vectoriser with the parameters then vectorise the training and testing data
max_features for how many stop words
ngram_range for the size of word combinations to use (in this case unigram (single word) and bigrams (two-word phrases))
min_df for how rare a word can be before being ignored
'''

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
xTrain = vectorizer.fit_transform(train['text'])
xTest = vectorizer.transform(test['text'])

'''
Initialise the classifier
keep the weights balanced
penalty to L1 so so its more strict on features
low C for the flexibility for better generalization
'''
cls = LinearSVC(class_weight='balanced', penalty='l1', dual=False, C=1.8)

# OR using Voting Classifier to combine multiple models and using soft voting for avg probabilities

model = VotingClassifier(estimators=[
    ('lr', LogisticRegression(class_weight='balanced', max_iter=2000)),
    ('dt', RandomForestClassifier(class_weight='balanced',n_estimators=200, random_state=42)),
    ('svc', SVC(probability=True))
], voting='soft')


# Set the output training and testing values
yTrain = train['sentiment']
yTest = test['sentiment']

# Train the classifier and predict using the testing values
model.fit(xTrain, yTrain)

yPred = model.predict(xTest)

# Evaluate the model and compare the predicted and tested values
print(classification_report(yTest, yPred))
print("Accuracy:", round(accuracy_score(yTest, yPred) * 100, 2), "%")

# Predict the user input whether its positive/neutral/negative
input = input("Enter text for sentiment analysis: ")
inputText = preprocess_text(input)
inputVectorise  = vectorizer.transform([inputText])
prediction = model.predict(inputVectorise)
print("This is predicted as", prediction[0])

