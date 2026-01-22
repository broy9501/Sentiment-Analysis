import random
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, TextVectorization, Embedding, LSTM, Bidirectional

# Read data
data = pd.read_csv("intentEmotion.csv")

tags, emotions, texts = [], [], []

# Get data and assign the values
for intent in data["intent"].values.astype(str) :
    tags.append(intent)

for emotion in data['emotion'].values:
    emotions.append(emotion)

for text in data['text'].values:
    texts.append(text)

# Sort the intents and remove duplicates to become unique
uniqueTags = sorted(set(tags))
tagByIndex = {tag: index for index, tag in enumerate(uniqueTags)}
y = [tagByIndex[i] for i in tags]

# Create vectoriser
vectoriser = TextVectorization(max_tokens=2000, output_mode='int', output_sequence_length=20)
vectoriser.adapt(texts)

# Build model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(), dtype=tf.string),
    vectoriser,
    Embedding(input_dim=2000, output_dim=32),
    # x = GlobalAveragePooling1D()(x)
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(uniqueTags), activation='softmax')
])

# Complile and train the model then save it
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

texts = np.array(texts, dtype=np.object_)
y = np.array(y, dtype=np.int32)

model.fit(texts, y, epochs=100, batch_size=8, validation_split=0.2, shuffle=True)

model.save('emotion_intent_model.h5')

print("Model trained and saved as 'chat_intent_model.h5'")

# With the user input and get predictions with the model
def predict_intent(inputText):
    inputT = np.array([inputText], dtype=np.object_)
    predictions = model.predict(inputT)[0]
    index = int(np.argmax(predictions))
    score = np.max(predictions) * 100
    # print(f"Predicted score: {score}")

    # Get most predicted intent tag
    mostPredicted = []
    for x, i in data.iterrows():
        if i['intent'] == uniqueTags[index]:
            mostPredicted.append(i["emotion"])
        if mostPredicted:
                # print(f"Predicted tag: {predicted_tag}")
                predictedIntent = uniqueTags[index]
                # most = max(set(mostPredicted), key=mostPredicted.count)
                # print(most)

    return predictedIntent, score

# predicted_tag, score = predict_intent("I feel more confident about this topic now")
# mostPredicted = []
# for index, i in data.iterrows():
#     if i['intent'] == predicted_tag:
#         mostPredicted.append(i["emotion"])
# if mostPredicted:
#         print(f"Predicted tag: {predicted_tag}")
#         most = max(set(mostPredicted), key=mostPredicted.count)
#         print(most)
    # print(Counter(mostPredicted).most_common(1)[0][0])