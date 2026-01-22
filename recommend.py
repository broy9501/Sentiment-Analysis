from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from sentence_transformers import util, SentenceTransformer

import pandas as pd

from emotionDetect import detect_emotion
from intentEmotion import predict_intent
from mappingCheck import mapCreate

# Get user input
user_input = input("How are you feeling?: ")

# Read Mood -> Action dataset
data = pd.read_csv("EmotionAction.csv")

# Get detected emotion and the intent from classifiers
emotion = detect_emotion(user_input)[0]
intent, score = predict_intent(user_input)

# print("emotion: ", emotion)
# print("intent: ", intent)

# INTENT_CATEGORY_MAP = {
#     "greet": ["maintenance"],
#     "farewell": ["maintenance"],
#     "thank": ["maintenance"],
#     "apologize": ["reflection"],

#     "inform": ["maintenance"],
#     "affirm": ["maintenance"],
#     "deny": ["reflection"],

#     "question": ["reflection", "focus"],
#     "request": ["focus"],

#     "seek_help": ["rest", "reflection"],
#     "seek_advice": ["reflection", "focus"],

#     "express_difficulty": ["rest", "focus"],
#     "express_frustration": ["rest", "physical"],
#     "express_uncertainty": ["reflection", "focus"],

#     "motivation_seeking": ["motivation"],
#     "express_progress": ["growth", "leverage"],
#     "express_confidence": ["challenge", "growth"],

#     "self_reflection": ["reflection"],
#     "express_relief": ["maintenance"],

#     "planning": ["focus", "growth"],
#     "decision_making": ["reflection", "focus"]
# }

# Getting intent to category map from function
INTENT_CATEGORY_MAP = mapCreate()

# Get intent, category values that matches with classifier's intent
filter_category = INTENT_CATEGORY_MAP.get(intent, ["reflection"])

# print(filter_category)

# actions = data[(data["Category"].isin(filter_category)) & (data["Emotion"] == emotion)]["Action"].tolist()
# reasons = []

# for action in actions:
#     reasons = data[(data["Category"].isin(filter_category)) & (data["Emotion"] == emotion) & (data["Action"] == action)]["Reason"].tolist()

# Filter out the data
filtered = data[(data["Category"].isin(filter_category)) & (data["Emotion"] == emotion)].reset_index(drop=True)

# Get the actions and reasons from the filtered data
actions = filtered["Action"].tolist()
reasons = filtered["Reason"].tolist()

# print(actions)
# print("\n\n")
# print(reasons)


if len(actions) > 0:
    # vectoriser = TfidfVectorizer()
    # all_texts = actions + [user_input]
    # vectoriser.fit(all_texts)
    # actionVector = vectoriser.transform(actions)
    # userInputVector = vectoriser.transform([user_input])
    # cosineSimilarities = cosine_similarity(userInputVector, actionVector)[0]

    # Embed user input and the actions
    model = SentenceTransformer('all-MiniLM-L6-v2')
    userInputEncode = model.encode(user_input, convert_to_tensor=True)
    actionEncode = model.encode(actions, convert_to_tensor=True)

    # Calculate the similarities between the input and actions
    cosineSimilarities = util.cos_sim(userInputEncode, actionEncode)[0]

    # get the index of the highest value and get the best action and reason
    index = cosineSimilarities.argmax()
    best_action = actions[index.item()]
    best_reason = reasons[index.item()]

    # Display the results
    #if "cosineSimilarities" in locals():
    print(cosineSimilarities)
    print("\n\n")
    print("What you should do (action): ", best_action)
    print("Why you should do it (reason): ", best_reason)
elif len(actions) == 0:
    best_action = "Take a short pause and reflect on one small next step"
    best_reason = "No exact match was found, so a safe reflective action was suggested"
else:
    print("error")