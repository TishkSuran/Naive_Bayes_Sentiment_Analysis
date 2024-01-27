import re 
from collections import Counter
from math import log
from data import documents, labels
from sklearn.model_selection import train_test_split

print("Data set size:", len(documents))


class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.word_probabilities = {}

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        stop_words = set(['the', 'this', 'i', 'as', 'its', 'it', 'and', 'is', 'for', 'in', 'of', 'to', 'with', 'on', 'at', 'by', 'an', 'product'])
        words = [word for word in text.split() if word not in stop_words]
        return ' '.join(words)

    def tokenise(self, text):
        return text.split()

    def count_words(self, documents, labels):
        word_counts = {}
        label_counts = Counter(labels)
        
        for doc, label in zip(documents, labels):
            words = self.tokenise(self.preprocess_text(doc))
            for word in words:
                if word not in word_counts:
                    word_counts[word] = {}
                if label not in word_counts[word]:
                    word_counts[word][label] = 1
                else:
                    word_counts[word][label] += 1
        
        return(word_counts, label_counts)

    def train(self, documents, labels, alpha):
        word_counts, label_counts = self.count_words(documents, labels)
        total_documents = len(documents)
        
        for label, count in label_counts.items():
            self.class_probabilities[label] = count / total_documents
            
        for word, label_counts in word_counts.items():
            self.word_probabilities[word] = {}
            for label, count in label_counts.items():
                total_words_in_label = sum(label_counts.values())
                self.word_probabilities[word][label] = (count + alpha) / total_words_in_label
                
    def predict(self, document):
        document = self.preprocess_text(document)
        words = self.tokenise(document)
        
        scores = {label: log(prob) for label, prob in self.class_probabilities.items()}
        
        for word in words:
            if word in self.word_probabilities:
                for label, prob in self.word_probabilities[word].items():
                    scores[label] += log(prob)
                    
        prediction = max(scores, key=scores.get)
        return prediction
    
def evaluate_accuracy(classifier, X_test, y_test):
    correct_predictions = 0
    total_predictions = len(X_test)

    for doc, true_label in zip(X_test, y_test):
        predicted_label = classifier.predict(doc)
        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {round(accuracy,4)}")


X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=20)
classifier = NaiveBayesClassifier()
classifier.train(X_train, y_train, alpha=1)
evaluate_accuracy(classifier, X_test, y_test)

while True:
    test_document = input("Please write a review of the product: ")
    prediction = classifier.predict(test_document)
    print(f"Predicted sentiment: {prediction}")