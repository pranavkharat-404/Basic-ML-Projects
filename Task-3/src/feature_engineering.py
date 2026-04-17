from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(train_texts, test_texts):

    tfidf = TfidfVectorizer(
        max_features=3000,  
        stop_words='english' 
    )


    X_train = tfidf.fit_transform(train_texts)

    X_test = tfidf.transform(test_texts)

    return X_train, X_test, tfidf