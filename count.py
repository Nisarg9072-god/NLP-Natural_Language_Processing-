
from sklearn.feature_extraction.text import TfidfVectorizer

document =[
    'I love pizza',
    'pizza is the best',
    'I love pasta',
    'pasta is great'
]

vectorizer = TfidfVectorizer()
X= vectorizer.fit_transform(document)
print("Vocab:", vectorizer.get_feature_names_out())
print("Matrix:",X.toarray())
