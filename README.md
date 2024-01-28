### What is the Naive Bayes Classifier? 
Before we dissect this project, it is worth stating the formal definition of the Naive Bayes Classifier, the algorithm used for sentiment analysis. The Naive Bayes Classifier is a probabilistic machine learning algorithm based on Bayes' Theorem. It is employed primarily for classification tasks, as it is in this project, where the goal is to categorise input data into predefined classes or categories. Despite its simplicity and the "naive" assumption, the Naive Bayes Classifier performs well in practice and is computationally efficient.
<br>

### What is Bayes' Theorem?
As previously mentioned, the Naive Bayes Classifier is based on Bayes' Theorem, thus it is a logical step to first explain Bayes Theorem for those who are not familiar. Bayes' Therom is a fundamental principle in probability theory. It provides a way to update probabilities based on new evidence. The therom can be expressed mathematically as follows:

$\ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \$

Where:
<ul>
  <li>$P(A|B)$ is the probability of event A given that event B has occured.</li>
  <li>$P(B|A)$ is the probability of event B given that event A has occured.</li>
  <li>$P(A)$ is the prior probability of event A.</li>
  <li>$P(B)$ is the prior probability of event B.</li>
</ul>
<br>

### Example Usage of Bayes Therom:
Suppose there is a rare disease, let's call it Disease X, that affects 1 in 10,000 people. A new test has been developed to detect the presence of Disease X. The text is 99% accurate, meaning that it correctly identifies individuals with the disease 99% of the time, and it has a 1% false positive rate, meaning it incorrectly identifies healthy individuals as having the disease 1% of the time.

Now, let's say you take the test and it comes back positive. What is the probability that you actually have the disease? We can use Bayes' Therom to calculate this, let...

<ul>
  <li><strong>A</strong> be the event that you have the disease (Disease X).</li>
  <li><strong>B</strong> be the event that the test is positive.</li>
</ul>

Remember the formula for Bayes' Theorem is:

$\ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \$

Where in this case:
<ul>
  <li>$P(B|A)$ is the probability of obtaining a positive test result, given the person actually has the disease. In this example, the test is said to be 99% accurate, meaning it is correct 99% of the time. Thus the probability of the test being positive in the event you have the disease is 0.99</li>
  <li>$P(A)$ = 0.0001, since 1 in 10,000 people have the disease initially.</li>
  <li>$P(B)$ is the probability of a positive test result. So we must take into account both the true positives and the false positives. Therefor $P(B) = (0.99 * 0.0001) + (0.01 * 0.9999).$</li>
</ul>

Thus the probability of having the disease given that the test result is positive is in fact only ≈0.0989%.
<br>
<br>

### Application of Bayes' Theorem in Classification:
Now that we have a general understanding of Bayes' Theorem, let's see how it seamlessly integrates into the Naive Bayes Classifier for classification tasks. In the context of sentiment analysis, the goal is to classify input data into predefined classes or categories such as positive or negative sentiment.

In the Naive Bayes Classifier, we apply Bayes' Theorem to calculate the probability of a particular class given the input features. The "naive" assumption in this classifier is that the features are conditionally independent, given the class. Despite the simplification, and as we will see, the Naive Bayes Classifier often performs quite well in practice.
<br>
<br>

### Example Usage of Naive Bayes Classification:
The best way to help you understand the Naive Bayes Classification system is through an example. In this example, we will be classifying emails as either spam or not spam. 

Let's say we have a new email with the following features:
<ol>
  <li>$X₁$: The presence or absence of the word "offer".</li>
  <li>$X₂$: The presence or absence of the word "money".</li>
</ol>

And we have two classes:

<ol>
  <li>$C₁$: Spam</li>
  <li>$C₂$: Not Spam</li>
</ol>

Now lets say we have a new email with the following features
<ul>
  <li>$X₁$: True, contains the word "offer".</li>
  <li>$X₂$: False, does not contain the word "money".</li>
</ul>

We want to classify this email as either spam or not spam using Naive Bayes.

<strong>Step 1: Prior Probabilities, $P(C₁) and P(C₂):$</strong><br>
Suppose in our training data, 30% of emails are spam, $P(C₁) = 0.3$, and 70% are not spam, $P(C₂) = 0.7$.

<strong>Step 2: Likelihoods, $P(X₁|C_i)$ and $P(X₂|C_i)$:</strong>

We calculate the likelihoods based on our training data.
- $P(X₁=\text{True}|C₁)$ might be 0.8 (80% of spam emails contain the word "offer").
- $P(X₂=\text{False}|C₁)$ might be 0.2 (20% of spam emails do not contain the word "money").
- $P(X₁=\text{True}|C₂)$ might be 0.1 (10% of non-spam emails contain the word "offer").
- $P(X₂=\text{False}|C₂)$ might be 0.9 (90% of non-spam emails do not contain the word "money").

<strong>Step 3: Applying Naive Bayes:</strong><br>

Calculate the unnormalised probability for each class:<br>

For $C_1$:<br>
$P(C_1) \cdot P(X_1=\text{True}|C_1) \cdot P(X_2=\text{False}|C_1) = 0.3 \cdot 0.8 \cdot 0.2 = 0.048$<br>

For $C_2$:<br>
$P(C_2) \cdot P(X_1=\text{True}|C_2) \cdot P(X_2=\text{False}|C_2) = 0.7 \cdot 0.1 \cdot 0.9 = 0.063$<br>

Normalise the probabilities so they sum to 1.<br>

The class with the highest probability is the predicted class. If the probability for $C_1$ is higher, we classify the email as spam; otherwise, it's not spam. In this case, if an email is received with the word "offer" but not "money", it will be classified as not spam.
<br>
<br>
### Naive Bayes Sentiment Classifier:
Now that you know what Bayes' Theorem is as well as it's application in classification, we can now finally start breaking down the naive bayes sentiment classifier script. 
<br>
<br>
### Defining the Naive Bayes' Classifier Class:
```python
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.word_probabilities = {}
```
This Python script implements a Naive Bayes classifier for sentiment analysis. The <strong>'NaiveBayesClassifier"</strong> class is designed to analyse and predict sentiment labels based on a given set of training data. This class is initialised with empty dictionaries for <strong>'class_probabilities'</strong> and <strong>'word_probabilities'</strong>. 
<br>
<br>
### Pre-Processing Text:
```python
    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        stop_words = set(['the', 'this', 'i', 'as', 'its', 'it', 'and', 'is', 'for', 'in', 'of', 'to', 'with', 'on', 'at', 'by', 'an', 'product'])
        words = [word for word in text.split() if word not in stop_words]
        return ' '.join(words)
```
<br>
<br>

### Tokenising the Text: 
The <strong>'preprocess_text'</strong> method takes a raw text input and performs several preprocessing steps to prepare it for sentiment analysis. It first removes all non alphanumerical characters and converts the entire text to lower case. It then eliminates any common stop words, such as articles and prepositions. The processed text is then tokenised into individual words, and a final string is constructed by joining these words. The output is a cleaned and normalised representation of input text.
<br>
<br>
```python
    def tokenise(self, text):
        return text.split()
```
This <strong>'tokenise'</strong> method simply takes text input and tokenises it by splitting it into individual words using the <strong>'split()</strong> method which seperates the input text into a list of words based on whitespace.
<br>
<br>
### Counting the Occurrences of Words:
```python
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
```
This method is responsible for tallying the occurrences of words within the training documents, categorised by their associated sentiment labels. This function initialises two dictionaries, <strong>'word_counts'</strong> and <strong>'label_counts</strong> to store the counts of each label and the total counts for each label, respectively. The method then iterates through each document-label pair in the training data, pre processes and tokenises the document using the previous methods and updates the <strong>'word_counts'</strong> dictionary. For each word encountered, the method ensures that the word is presnet in the directory and increments its count for the corresponding label. This process establishes a count of how often each word appears in doucmnets associated with different sentiment labels, forming the basis for following probability calculations during the training phase of the Naive Bayes classifier. 
<br>
<br>
### Training the Classifier:
```python
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
```
This <strong>predict</strong> method takes the document as its input, preprocesses it and tokenises it using the methods defined above. It then initialises scores for each of the sentiment labels based on logarithms and class probabilities. For each word in the tokenised document, it will update the scores using logarithms of word probabilities associated with each label. The method then selects the sentiment lavel with the highest score as the predicted sentiment for the input document. Logs are used to help with numerical stability, particularly to mitigate issues associated with very small probabilities. 
<br>
<br>
### The Remainder of the Code 
```python
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
```
The remaining code is dedicated to assessing the model's accuracy through the separation of documents and labels into training and testing datasets. The concluding segment establishes a user interaction loop, enabling users to engage with the script and input product reviews for sentiment prediction.
