documents = [
    "I love this product!",
    "This is terrible.",
    "Amazing experience!",
    "Waste of money.",
    "Great value for the price.",
    "This product is bad!",
    "Terrible customer service.",
    "Disappointing quality.",
    "Not worth it.",
    "I am satisfied with my purchase.",
    "This is the best product ever!",
    "Absolutely horrible.",
    "Outstanding performance.",
    "I regret buying this.",
    "Highly recommended!",
    "I would never buy this again.",
    "Unbelievably good.",
    "Worst purchase ever.",
    "Impressed with the quality.",
    "Top-notch craftsmanship.",
    "Complete waste of time.",
    "Fantastic product!",
    "Not up to the mark.",
    "I can't live without it.",
]

labels = [
    "positive", "negative", "positive",
    "negative", "positive", "negative",
    "negative", "negative", "negative",
    "positive", "positive", "negative",
    "positive", "negative", "positive",
    "negative", "positive", "negative",
    "negative", "positive", "negative",
    "positive", "negative", "positive"
]

additional_documents = [
    "Excellent choice!",
    "Regrettable decision.",
    "Thrilled with my purchase.",
    "Huge disappointment.",
    "Well worth the investment.",
    "This is a scam.",
    "Superb quality and service.",
    "Awful experience overall.",
    "Couldn't be happier!",
    "Do not recommend.",
    "Impressive performance.",
    "Complete rip-off.",
    "Satisfied customer here!",
    "Dreadful product.",
    "Highly satisfied with the value.",
    "Stay away from this!",
    "Absolutely fantastic.",
    "Never again!",
    "Top-tier product!",
    "Poorly made and overpriced.",
    "A must-have!",
    "Disastrous purchase.",
    "I love the features.",
    "Disgusted with the quality."
]

additional_labels = [
    "positive", "negative", "positive",
    "negative", "positive", "negative",
    "positive", "negative", "positive",
    "negative", "positive", "negative",
    "positive", "negative", "positive",
    "negative", "positive", "negative",
    "positive", "negative", "positive",
    "negative", "positive", "negative"
]

documents.extend(additional_documents)
labels.extend(additional_labels)
