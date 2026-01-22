import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text: str) -> str:
<<<<<<< HEAD
    if pd.isna(text):              # NaN / None
        return ""
    if not isinstance(text, str):  # float, int, etc.
=======
    if pd.isna(text):
        return ""
    if not isinstance(text, str): 
>>>>>>> 6abcd5023d8a49dcf459649730839afd9957307e
        text = str(text).strip()

    english_stopwords = set(stopwords.words('english'))
    finaltoken = [word for word in word_tokenize(text) if word.lower() not in english_stopwords]
    noPunctuation = [re.sub(r'[^\w\s]', '', word) for word in finaltoken]

    for x in noPunctuation:
        if x  == '':
            noPunctuation.remove(x)

    return ' '.join(noPunctuation)