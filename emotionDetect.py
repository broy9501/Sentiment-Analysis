import pandas as df
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sentence_transformers import SentenceTransformer

from preprocessing import preprocess_text

# Read data and preprocessing text column
data = df.read_csv("emotionLabels.csv")
data['text'] = data["text"].apply(preprocess_text)

# splitting data into training, testing, validation
train_val, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["emotion"])
train, val = train_test_split(train_val, test_size=0.50, random_state=42, stratify=train_val["emotion"])

# vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=2)
# trainX = vectorizer.fit_transform(train["text"])
# testX = vectorizer.transform(test["text"])


embedder = SentenceTransformer("all-mpnet-base-v2")

# Embed data and assign values
trainX = embedder.encode(train["text"].tolist())
valX = embedder.encode(val["text"].tolist())
testX = embedder.encode(test["text"].tolist())

# Create output variables
ytrain = train["emotion"]
yval = val["emotion"]
ytest = test["emotion"]


# Intialise voting classifier with soft voting
model = VotingClassifier(estimators=[
    ('lr', LogisticRegression(class_weight='balanced', max_iter=2000, C=2.0, solver="lbfgs")),
    # ('dt', RandomForestClassifier(class_weight='balanced',n_estimators=200, random_state=42)),
    ('svc', SVC(kernel="rbf", C=3, gamma="scale", probability=True, class_weight="balanced"))
], voting='soft')

# Train model
model.fit(trainX, ytrain)

# Get predicted values using test data
yPred = model.predict(testX)

# Evaluate the model and compare the predicted and tested values
print(classification_report(ytest, yPred))
print("Accuracy:", round(accuracy_score(ytest, yPred) * 100, 2), "%")

# Embed user input and get predicted emotion
def detect_emotion(inputTXT):
    # input = input("Enter text for sentiment analysis: ")
    inputText = preprocess_text(inputTXT)
    # inputVectorise  = vectorizer.transform([inputText])
    inputEmbed = embedder.encode(inputText)
    prediction = model.predict_proba([inputEmbed])[0]
    labels = model.classes_
    k=3
    index = np.argsort(prediction)[-k:][::-1]
    results = [labels[i] for i in index]
    # prediction = model.predict(inputVectorise)
    return results

# inputTXT = "I'm worried I won't do well"

# print("This is predicted as", detect_emotion(inputTXT)[0])