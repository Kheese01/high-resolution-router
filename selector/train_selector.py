import joblib
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X, y)

joblib.dump(clf, "model_weights/selector.pkl")
