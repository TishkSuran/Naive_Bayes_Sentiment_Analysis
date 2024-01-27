### What is the Naive Bayes Classifier? 
Before we dissect this project, it is worth stating the formal definition of the Naive Bayes Classifier, the algorithm used for sentiment analysis. The Naive Bayes Classifier is a probabilistic machine learning algorithm based on Bayes' Theorem. It is employed primarily for classification tasks, as it is in this project, where the goal is to categorise input data into predefined classes or categories. Despite its simplicity and the "naive" assumption, the Naive Bayes Classifier performs well in practice and is computationally efficient.

### What is Baye's Therom?
As previously mentioned, the Naive Bayes Classifier is based on Bayes' Therom, thus it is a logical step to first explain Bayes Therom for those who are not familiar. Bayes' Therom is a fundamental principle in probability theory. It provides a way to update probabilities based on new evidence. The therom can be expressed mathematically as follows:

$\ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \$

Where:
<ol>
  <li>$P(A|B)$ is the probability of event A given that event B has occured.</li>
  <li>$P(B|A)$ is the probability of event B given that event A has occured.</li>
  <li>$P(A)$ is the prior probability of event A.</li>
  <li>$P(B)$ is the prior probability of event B.</li>
</ol>
