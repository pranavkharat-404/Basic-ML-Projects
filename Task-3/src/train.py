from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_models(X_train, y_train):
    
    models = {}

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    models['SVM'] = svm

    return models