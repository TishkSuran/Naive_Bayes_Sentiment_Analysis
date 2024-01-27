### What is the Naive Bayes Classifier? 
Before we dissect this project, it is worth stating the formal definition of the Naive Bayes Classifier, the algorithm used for sentiment analysis. The Naive Bayes Classifier is a probabilistic machine learning algorithm based on Bayes' Theorem. It is employed primarily for classification tasks, as it is in this project, where the goal is to categorise input data into predefined classes or categories. Despite its simplicity and the "naive" assumption, the Naive Bayes Classifier performs well in practice and is computationally efficient.
<br>
### What is Baye's Therom?
As previously mentioned, the Naive Bayes Classifier is based on Bayes' Therom, thus it is a logical step to first explain Bayes Therom for those who are not familiar. Bayes' Therom is a fundamental principle in probability theory. It provides a way to update probabilities based on new evidence. The therom can be expressed mathematically as follows:

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

Remember the formula for Bayes' Therom is:

$\ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \$

Where in this case:
<ul>
  <li>$P(B|A)$ is the probability of obtaining a positive test result, given the person actually has the disease. In this example, the test is said to be 99% accurate, meaning it is correct 99% of the time. Thus the probability of the test being positive in the event you have the disease is 0.99</li>
  <li>$P(A)$ = 0.0001, since 1 in 10,000 people have the disease initially.</li>
  <li>$P(B)$ is the probability of a positive test result. So we must take into account both the true positives and the false positives. Therefor $P(B) = (0.99 * 0.0001) + (0.01 * 0.9999).$</li>
</ul>

