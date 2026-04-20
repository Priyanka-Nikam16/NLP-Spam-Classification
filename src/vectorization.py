from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(texts):
    vectorizer=TfidfVectorizer( max_features=5000,
                                ngram_range=(1,2)  #unigram+bigram
                               )
    X=vectorizer.fit_transform(texts)
    
    return X,vectorizer

    