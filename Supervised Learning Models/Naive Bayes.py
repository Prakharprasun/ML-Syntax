# Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gaussian Naive Bayes
gnb = GaussianNB()

# Fit the model
gnb.fit(X_train, y_train)

# Predict on the test data
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


# Import necessary libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample text data
texts = ["I love programming", "Python is amazing", "I hate bugs", "Debugging is fun", "I love machine learning"]
labels = [1, 1, 0, 0, 1]  # 1: Positive, 0: Negative

# Convert text data to numerical data (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Initialize Multinomial Naive Bayes
mnb = MultinomialNB()

# Fit the model
mnb.fit(X_train, y_train)

# Predict on the test data
y_pred = mnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


# Import necessary libraries
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample text data
texts = ["I love programming", "Python is amazing", "I hate bugs", "Debugging is fun", "I love machine learning"]
labels = [1, 1, 0, 0, 1]  # 1: Positive, 0: Negative

# Convert text data to binary features (word presence)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Initialize Bernoulli Naive Bayes
bnb = BernoulliNB()

# Fit the model
bnb.fit(X_train, y_train)

# Predict on the test data
y_pred = bnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
