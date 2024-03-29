negative_review_words = [
    'Inaccurate',
    'Unreliable',
    'Biased',
    'Confusing',
    'Slow',
    'Inefficient',
    'Flawed',
    'Ineffective',
    'Unpredictable',
    'Complicated',
    'Error-prone',
    'Unintuitive',
    'Limited',
    'Frustrating',
    'Buggy',
    'Unstable',
    'Overfitting',
    'Underfitting',
    'Inconsistent',
    'Poorly trained',
    'Outdated',
    'Resource-intensive',
    'Non-responsive',
    'Difficult to use',
    'Lacks transparency',
    'Hate',
]


positive_review_words = [
    'Accurate',
    'Reliable',
    'Unbiased',
    'Clear',
    'Fast',
    'Efficient',
    'Robust',
    'Effective',
    'Predictable',
    'Simple',
    'Error-free',
    'Intuitive',
    'Flexible',
    'Satisfying',
    'Stable',
    'Well-performing',
    'Generalizing',
    'Adaptable',
    'Consistent',
    'Well-trained',
    'Up-to-date',
    'Resource-efficient',
    'Responsive',
    'Easy to use',
    'Transparent',
    'Good',
]

long_negative_reviews = [
    "This product is a complete disaster. Inaccurate, unreliable, and a waste of money.",
    "Absolutely terrible. Slow, inefficient, and error-prone. Not worth the price.",
    "Avoid this at all costs. Biased, confusing, and frustrating.",
    "I regret buying this. Limited features and poor quality. Unstable and outdated.",
    "Worst purchase ever. The model is inconsistent, unreliable, and non-responsive.",
    "A complete rip-off. Overfitting and underfitting issues. Difficult to use and lacks transparency.",
    "I can't believe I wasted my time on this. Unpredictable and buggy. Stay away!",
    "This is a scam. Don't be fooled by the claims of value. It's a complete disappointment.",
    "Highly dissatisfied. Unintuitive, flawed, and poorly trained. Save your money.",
    "I wouldn't recommend this to my worst enemy. Confusing interface and limited functionality.",
    "Huge disappointment. Slow performance, non-responsive, and absolutely useless.",
    "Complete waste of time and resources. Inconsistent results and outdated algorithms.",
    "This product is a joke. Terrible customer service and absolutely no transparency.",
    "Dreadful experience. Overpriced and unreliable. Don't make the mistake I did.",
    "I've never been so disappointed. Unstable, non-responsive, and a complete letdown.",
    "This is the worst product I've ever encountered. Save yourself the trouble and avoid it.",
    "Disastrous purchase. The quality is poor, and it's not even close to being worth the money.",
    "I can't find a single positive thing to say about this product. Completely unusable.",
    "I wouldn't wish this product on my worst enemy. Avoid like the plague.",
    "I've had better experiences with free software. Limited features and constant errors.",
    "Stay far away from this product. Unpredictable behavior and terrible performance.",
    "I deeply regret spending my hard-earned money on this. Absolutely terrible in every way.",
    "I thought it couldn't get worse, but it did. Inconsistent, unreliable, and a total waste of time.",
    "This product is an embarrassment. I can't believe it's on the market. Save yourself the trouble and steer clear.",
    "Complete garbage. The features are a joke, and the performance is abysmal.",
    "I've never written a negative review before, but this product deserves it. Stay away!",
    "I wouldn't wish this on my worst enemy. Biased, buggy, and absolutely not worth the money.",
    "Terrible quality and terrible service. The developers should be ashamed of themselves.",
    "I can't express how disappointed I am with this product. Slow, unstable, and completely useless.",
    "This product is an insult to anyone who uses it. Unreliable, non-responsive, and a complete waste of time.",
    "I wish I could get a refund for the time I wasted on this. Absolutely horrendous.",
    "I'm convinced this product was designed to torture users. Unintuitive, frustrating, and a complete failure.",
    "Do yourself a favor and avoid this product at all costs. Inconsistent, unreliable, and a headache to use.",
    "I have never encountered a more useless product. Inaccurate, non-responsive, and an absolute disaster.",
    "I wouldn't wish this on my worst enemy. It's a complete nightmare to work with.",
    "Save your money and your sanity. This product is a disaster from start to finish.",
    "I have never been so frustrated with a product. It's a complete letdown in every way.",
    "This is the worst investment I've ever made. Unstable, unreliable, and not fit for purpose.",
    "I've never experienced such a lack of quality in a product. Inconsistent, buggy, and a complete waste of time.",
    "I wouldn't recommend this to my worst enemy. It's a complete disaster.",
    "I have nothing positive to say about this product. It's a complete failure in every aspect.",
    "I can't believe this product is still on the market. It's a disgrace to the industry.",
    "I've never felt compelled to leave a negative review until now. This product is an absolute nightmare.",
    "This product is an embarrassment. The developers should be ashamed of themselves for releasing such a disaster.",
    "I can't find a single redeeming quality in this product. It's a complete and utter failure.",
    "Stay far away from this product. It's a complete waste of time and money.",
    "I'm beyond disappointed with this product. It's a complete letdown in every way.",
    "I've never encountered a product so poorly designed and executed. It's a complete disaster."
]

long_positive_reviews = [
    "This product is an absolute delight. Accurate, reliable, and worth every penny.",
    "Incredible experience with this product! Fast, efficient, and error-free. A true gem.",
    "I can't express how satisfied I am with this purchase. Simple, intuitive, and a joy to use.",
    "This is a game-changer. Well-performing, adaptable, and consistently excellent.",
    "Outstanding product! Generalizing and adapting features make it a top-tier solution.",
    "I'm impressed with the quality and performance of this model. It's well-trained and up-to-date.",
    "This product is a must-have. Resource-efficient, responsive, and transparent.",
    "Absolutely fantastic! The craftsmanship and features are beyond expectations.",
    "I love everything about this product. It's flexible, satisfying, and stable.",
    "Thrilled with my purchase. The value is well worth the investment.",
    "Highly recommended! Superb quality and service. Couldn't be happier.",
    "This is an excellent choice. I'm thrilled with my purchase and its fantastic features.",
    "I am a satisfied customer. Regrettable decision? Absolutely not. It's a delight.",
    "Huge satisfaction with the value provided by this product. A top-tier solution.",
    "Stay away from negativity; this product is absolutely fantastic!",
    "Never again will I settle for less. This top-notch product exceeded my expectations.",
    "This product is a masterpiece. I'm highly satisfied with its quality and features.",
    "I can't imagine my life without this. It's an essential and fantastic solution.",
    "The features are mind-blowing. This product is an absolute game-changer.",
    "I'm impressed with the top-tier quality and service provided by this product.",
    "Absolutely thrilled with my purchase. This product is a must-have for everyone.",
    "Unbelievably good experience with this outstanding product. Highly recommended!",
    "I'm grateful for the fantastic features and performance of this top-notch product.",
    "This is the best investment I've ever made. Outstanding performance and quality.",
    "I can't express enough how delighted I am with this purchase. Fantastic all around.",
    "This product is top-tier in every aspect. Well worth the investment.",
    "I can't get enough of the superb quality and craftsmanship of this excellent product.",
    "Outstanding value for the price. This product has exceeded my expectations.",
    "I am beyond satisfied with the quality and performance of this fantastic product.",
    "I'm hooked on the features of this outstanding and reliable product. Absolutely love it.",
    "This product is a gem. Efficient, adaptable, and consistently performs at its best.",
    "I'm in awe of the simplicity and effectiveness of this top-notch product.",
    "The transparency and responsiveness of this product are truly commendable.",
    "I can't help but praise the well-trained and up-to-date model. Excellent investment.",
    "This product is a lifesaver. Resource-efficient, responsive, and easy to use.",
    "I'm grateful for the transparent and reliable features of this fantastic product.",
    "This product is a joy to use. The craftsmanship and quality are exceptional.",
    "Outstanding value for the price. This top-tier product is a delight to own.",
    "I'm amazed at the flexibility and stability of this excellent product. Highly recommended.",
    "Never settle for less when you can have the best. This product is top-notch.",
    "I'm thoroughly impressed with the performance and adaptability of this fantastic product.",
    "This product is a dream come true. Generalizing features make it a top-tier solution.",
    "Absolutely fantastic! The craftsmanship and performance are beyond expectations.",
    "This is an excellent choice. I'm thrilled with my purchase and its fantastic features.",
    "I am a satisfied customer. Regrettable decision? Absolutely not. It's a delight.",
    "Huge satisfaction with the value provided by this product. A top-tier solution.",
    "Stay away from negativity; this product is absolutely fantastic!",
    "Never again will I settle for less. This top-notch product exceeded my expectations.",
    "This product is a masterpiece. I'm highly satisfied with its quality and features.",
    "I can't imagine my life without this. It's an essential and fantastic solution.",
    "The features are mind-blowing. This product is an absolute game-changer.",
    "I'm impressed with the top-tier quality and service provided by this product.",
    "Absolutely thrilled with my purchase. This product is a must-have for everyone.",
    "Unbelievably good experience with this outstanding product. Highly recommended!",
    "I'm grateful for the fantastic features and performance of this top-notch product.",
    "This is the best investment I've ever made. Outstanding performance and quality.",
    "I can't express enough how delighted I am with this purchase. Fantastic all around.",
    "This product is top-tier in every aspect. Well worth the investment.",
    "I can't get enough of the superb quality and craftsmanship of this excellent product.",
    "Outstanding value for the price. This product has exceeded my expectations.",
    "I am beyond satisfied with the quality and performance of this fantastic product.",
    "I'm hooked on the features of this outstanding and reliable product. Absolutely love it.",
    "This product is a gem. Efficient, adaptable, and consistently performs at its best.",
    "I'm in awe of the simplicity and effectiveness of this top-notch product.",
    "The transparency and responsiveness of this product are truly commendable.",
    "I can't help but praise the well-trained and up-to-date model. Excellent investment.",
    "This product is a lifesaver. Resource-efficient, responsive, and easy to use.",
    "I'm grateful for the transparent and reliable features of this fantastic product.",
    "This product is a joy to use. The craftsmanship and quality are exceptional.",
    "Outstanding value for the price. This top-tier product is a delight to own.",
    "I'm amazed at the flexibility and stability of this excellent product. Highly recommended.",
    "Never settle for less when you can have the best. This product is top-notch.",
    "I'm thoroughly impressed with the performance and adaptability of this fantastic product.",
    "This product is a dream come true. Generalizing features make it a top-tier solution."
]

medium_negative_reviews = [
    "This product is unreliable and slow. Limited features and confusing interface. Definitely not worth the money.",
    "I'm disappointed with this purchase. The model is inconsistent and error-prone. It lacks transparency and is difficult to use.",
    "Average at best. Non-responsive and outdated. The performance is subpar, and I wouldn't recommend it.",
    "Not the worst, but far from good. Unintuitive design and frustrating to use. It's a letdown for the price.",
    "Could be better. Inefficient and buggy. The features are lacking, and it feels like a waste of time.",
    "A mixed bag. The model is biased and poorly trained. Overfitting issues make it unpredictable.",
    "This product is a letdown. Unstable and inconsistent results. I regret buying it.",
    "Not living up to expectations. Slow and resource-intensive. The quality is substandard.",
    "I expected more. Confusing and error-prone. The overall experience is frustrating.",
    "This is below average. The product is outdated and unreliable. Limited functionality and poor performance."
]

medium_positive_reviews = [
    "Great value for the price. The product is accurate and efficient. I'm satisfied with my purchase.",
    "Amazing experience! Fast and effective. The model is well-trained and performs predictably.",
    "Worth the investment. Reliable and stable. The features are flexible and satisfying to use.",
    "Impressed with the quality. The model is adaptable and up-to-date. Well worth the money.",
    "Highly recommended! Clear and intuitive. This product is a game-changer.",
    "Top-notch performance. Accurate and responsive. The craftsmanship is outstanding.",
    "Absolutely fantastic! The features are transparent and easy to use. A must-have.",
    "Thrilled with my purchase. Well-performing and robust. This product is top-tier.",
    "Excellent choice! Generalizing and adapting features make it a top-tier solution.",
    "I can't live without it. Satisfying and resource-efficient. The value provided is exceptional."
]

documents = []
labels = []


documents.extend(negative_review_words)
labels.extend(["negative"] * len(positive_review_words))

documents.extend(positive_review_words)
labels.extend(["positive"] * len(negative_review_words))

documents.extend(long_negative_reviews)
labels.extend(["negative"] * len(long_negative_reviews))

documents.extend(long_positive_reviews)
labels.extend(["positive"] * len(long_positive_reviews))

documents.extend(medium_negative_reviews)
labels.extend(["negative"] * len(medium_negative_reviews))

documents.extend(medium_positive_reviews)
labels.extend(["positive"] * len(medium_positive_reviews))
