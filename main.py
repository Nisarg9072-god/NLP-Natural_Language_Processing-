from enum import unique
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
import string


df =pd.read_csv("train.txt",sep=";",header=None,names=['text','emotion'])
print(df.head())

# print(df.isnull().sum())
# print(df['emotion'].unique())
# print(df['emotion'].value_counts())
# sns.countplot(x='emotion',data=df)
# plt.show()

uniques_emotion =df['emotion'].unique()
print(uniques_emotion)

emotion_number={}
i=0
#this for  TO SET THE NO TO THE EMOTIONS
for emo in uniques_emotion:
    
    emotion_number[emo]=i
    i+=1
# map the emotions to the numbers
df['emotion'] = df['emotion'].map(emotion_number)

#CONVERT TO THE LOWER CASE TO ALL THE DATA
df['text'] = df['text'].apply(lambda x:x.lower())

#REMOVE ALL THE PUNCHUATION
def remove_punch(txt):
    return txt.translate(str.maketrans('','',string.punctuation))


df['text'] = df['text'].apply(remove_punch)

#REMOVE ALL THE NON ALPHABETICAL CHARACTER
def remove_non(txt):
    new =''
    for i in txt:
        if not i.isdigit():
            new =new + i
    return new

df ['text']=df['text'].apply(remove_non)

#REMOVE THE MENTION AND URL FROM THE DATA
df['text'] = df['text'].apply(lambda x:x.replace('@user',''))
df['text'] = df['text'].apply(lambda x:x.replace('URL',''))


#TO REMOVE THE STOP WORDS WE WILL USE THE NLT (NATURAL LANGUAGE TOOLKIT) AND SPACY

#USING THE NLT
import nltk
'''
now there is your data with messages and that is know as the corpus

#CORPUS
#A COLLECTION OF TEXT . THINK OF IT LIKE A DATASET OF TEXT
#A GROUP OF WORDS THAT FORMS THE MEANING FULL SENTENCES
#TOKENIZATION SENTENCE SPLITING INTO WORDS 
#WORD TOKENIZATION MEANING SPLITING INTO WORDS 
#SENTENCE TOKENIZATION MEANING SPLITING INTO SENTENCES

'''

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_word=set(stopwords.words('english'))


#REMOVING THE STOP WORDS
#WHATEVWEVER 

def remove(txt):
    words= word_tokenize(txt)
    cleaned =[]
    for i in words:
        if i not in stop_word:
            cleaned.append(i)

    return " ".join(cleaned)

df['text']=df['text'].apply(remove)
print(df['text'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'],df['emotion'] , test_size=0.33, random_state=42)  


#USING THE TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

bow_vectorizer=CountVectorizer()
bow_vectorizer.fit(X_train)
X_train_bow=bow_vectorizer.transform(X_train)
X_test_bow=bow_vectorizer.transform(X_test)

tfidf_vectorizer=TfidfVectorizer()
tfidf_vectorizer.fit(X_train)
X_train_tfidf=tfidf_vectorizer.transform(X_train)
X_test_tfidf=tfidf_vectorizer.transform(X_test)




from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# using bow
model=MultinomialNB()
model.fit(X_train_bow,y_train)

# using tf-idf
model2=MultinomialNB()
model2.fit(X_train_tfidf,y_train)


# using bow
pred=model.predict(X_test_bow)
print("accuracy_score",accuracy_score(y_test,pred))
print("classification_report",classification_report(y_test,pred))

# using tf-idf
pred2=model2.predict(X_test_tfidf)
print("accuracy_score",accuracy_score(y_test,pred2))
print("classification_report",classification_report(y_test,pred2))
