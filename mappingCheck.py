import pandas as pd
from collections import defaultdict

# Read the csv data
data1 = pd.read_csv("EmotionAction.csv")
data2 = pd.read_csv("intentEmotion.csv")

# Create a dictionary of every intent to emotion map
# print("Intent to Emotion")
emoTointent = (data2
    .groupby("intent")["emotion"]
    .apply(lambda x: set(x))
    .to_dict()               
)

# for key, value in emoTointent.items():
#     print(f"{key} : {value}\n")

# print("\n\n\n\n\n")

# Create a dictionary of every category to emotion map
catToemo = (
    data1
    .groupby("Category")["Emotion"]
    .apply(lambda x: set(x))
    .to_dict()
)

# print("Category to Emotion")
# for key, value in catToemo.items():
#     print(f"{key} : {value}\n")

# intentTocat = defaultdict(set)
# for catToemoKey, catToemoValue in catToemo.items():
#     for emoTointentKey, emoTointentValue in emoTointent.items():
#         # for c in catToemoValue:
#         #     for e in emoTointentValue:
#         #         if e == c:
#         #             intentTocat[emoTointentKey] = catToemoKey
#         if catToemoValue & emoTointentValue:
#             intentTocat[emoTointentKey].add(catToemoKey)

# for key, value in intentTocat.items():
#     intentTocat[key] = list(value)

# Map the intent with categories using the emotions
def mapCreate():
    intentTocat = defaultdict(set)
    for catToemoKey, catToemoValue in catToemo.items():
        for emoTointentKey, emoTointentValue in emoTointent.items():
            # for c in catToemoValue:
            #     for e in emoTointentValue:
            #         if e == c:
            #             intentTocat[emoTointentKey] = catToemoKey
            if catToemoValue & emoTointentValue:
                intentTocat[emoTointentKey].add(catToemoKey)

    for key, value in intentTocat.items():
        intentTocat[key] = list(value)
    
    return intentTocat


# print("\n\n\n\n\n")

# map = mapCreate()
# print("Intent to Category")
# for key, value in map.items():
#     print(f"{key} : {value}\n")