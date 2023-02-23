dataset_name_to_metric = {
    "bbcnews": "accuracy",
    "sst2": "accuracy",
    "imdb_genre_prediction": "roc_auc",
    "sst5": "accuracy",
    "product_sentiment_machine_hack": "accuracy",
    "google_qa_answer_type_reason_explanation": "r2",
    "google_qa_question_type_reason_explanation": "r2",
    "bookprice_prediction": "r2",
    "goemotions": "jaccard",
    "jc_penney_products": "r2",
    "fake_job_postings2": "roc_auc",
    "data_scientist_salary": "accuracy",
    "news_popularity2": "r2",
    "women_clothing_review": "r2",
    "ae_price_prediction": "r2",
    "news_channel": "accuracy",
    "jigsaw_unintended_bias100K": "roc_auc",
    "ames_housing": "r2",
    "mercedes_benz_greener": "r2",
    "mushroom_edibility": "accuracy",
    "amazon_employee_access_challenge": "roc_auc",
    "naval": "r2",
    "sarcos": "r2",
    "protein": "r2",
    "adult_census_income": "accuracy",
    "otto_group_product": "accuracy",
    "santander_customer_satisfaction": "accuracy",
    "amazon_employee_access": "roc_auc",
    "numerai28pt6": "accuracy",
    "bnp_claims_management": "accuracy",
    "allstate_claims_severity": "r2",
    "santander_customer_transaction": "accuracy",
    "connect4": "accuracy",
    "forest_cover": "accuracy",
    "ieee_fraud": "accuracy",
    "porto_seguro_safe_driver": "accuracy",
    "walmart_recruiting": "accuracy",
    "poker_hand": "accuracy",
    "higgs": "accuracy",
}


dataset_name_to_repo_name = {
    "ames_housing": "Ames Housing Benchmarks",
    "mercedes_benz_greener": "Mercedes Benz Greener Benchmarks",
    "naval": "Naval Benchmarks",
    "adult_census_income": "Adult Census Income Benchmarks",
    "sarcos": "Sarcos Benchmarks",
    "protein": "Protein Benchmarks",
    "numerai28pt6": "Numerai28pt6 Benchmarks",
    "mushroom_edibility": "Mushroom Edibility Benchmarks",
    "amazon_employee_access_challenge": "Amazon Employee Access Challenge Benchmarks",
    "forest_cover": "Forest Cover Benchmarks",
    "santander_customer_satisfaction": "Santander Customer Satisfaction Benchmarks",
    "connect4": "Connect4 Benchmarks",
    "allstate_claims_severity": "Allstate Claims Severity Benchmarks",
    "santander_customer_transaction": "Santander Customer Transaction Benchmarks",
    "otto_group_product": "Otto Group Product Benchmarks",
    "bbcnews": "BBC News Benchmarks",
    "sst2": "SST2 Benchmarks",
    "imdb_genre_prediction": "IMDB Genre Prediction Benchmarks",
    "sst5": "SST5 Benchmarks",
    "product_sentiment_machine_hack": "Product Sentiment Machine Hack Benchmarks",
    "bookprice_prediction": "Book Price Prediction Benchmarks",
    "goemotions": "GoEmotions Benchmarks",
    "jc_penney_products": "JCPenny Products Benchmarks",
    "fake_job_postings2": "Fake Job Postings Benchmarks",
    "data_scientist_salary": "Data Scientist Salary Benchmarks",
    "news_popularity2": "News Popularity Benchmarks",
    "women_clothing_review": "Women Clothing Review Benchmarks",
    "ae_price_prediction": "AE Price Prediction Benchmarks",
    "news_channel": "News Channel Benchmarks",
    "jigsaw_unintended_bias100K": "Jigsaw Unintended Bias 100K Benchmarks",
    "ieee_fraud": "IEEE Fraud Benchmarks",
    "higgs": "Higgs Benchmarks",
    "titanic": "Titanic Benchmarks",
    "ohsumed_7400": "OHSUMED 7400 Benchmarks",
    "reuters_r8": "Reuters R8 Benchmarks",
    "mercari_price_suggestion100K": "Mercari Price Suggestion 100K Benchmarks",
    "melbourne_airbnb": "Melbourne AirBnb Benchmarks",
    "yelp_review_polarity": "Yelp Review Polarity Benchmarks",
    "yelp_reviews": "Yelp Reviews Benchmarks",
    "amazon_review_polarity": "Amazon Review Polarity Benchmarks",
    "amazon_reviews": "Amazon Reviews Benchmarks",
    "agnews": "AG News Benchmarks",
    "california_house_price": "California House Price Benchmarks",
    "ethos_binary": "Ethos Binary Benchmarks",
    "wine_reviews": "Wine Reviews Benchmarks",
    "yahoo_answers": "Yahoo Answers Benchmarks",
    "google_qa_answer_type_reason_explanation": "Google QAA Benchmarks",
    "google_qa_question_type_reason_explanation": "Google QAQ Benchmarks",
    "bnp_claims_management": "BNP Claims Management",
    "creditcard_fraud": "Credit Card Fraud Benchmarks",
    "imbalanced_insurance": "Imbalanced Insurance Benchmarks",
    "walmart_recruiting": "Walmart Recruiting Benchmarks",
}

dataset_name_to_use_case = {
    "ames_housing": "real estate",
    "mercedes_benz_greener": "industrial",
    "naval": "industrial",
    "adult_census_income": "income prediction",
    "sarcos": "robotics",
    "protein": "biology",
    "numerai28pt6": "financial services",
    "mushroom_edibility": "biology",
    "amazon_employee_access_challenge": "human resources",
    "forest_cover": "biology",
    "santander_customer_satisfaction": "financial services",
    "connect4": "gaming",
    "allstate_claims_severity": "insurance",
    "santander_customer_transaction": "sentiment analysis",
    "otto_group_product": "retail",
    "bbcnews": "topic classification",
    "sst2": "sentiment analysis",
    "imdb_genre_prediction": "topic classification",
    "sst5": "sentiment analysis",
    "product_sentiment_machine_hack": "sentiment anlysis",
    "bookprice_prediction": "retail",
    "goemotions": "sentiment analysis",
    "jc_penney_products": "retail",
    "fake_job_postings2": "fraud detection",
    "data_scientist_salary": "human resources",
    "news_popularity2": "media",
    "women_clothing_review": "sentiment analysis",
    "ae_price_prediction": "retail",
    "news_channel": "media",
    "ieee_fraud": "fraud detection",
    "higgs": "scientific",
    "titanic": "insurance",
    "ohsumed_7400": "medical",
    "reuters_r8": "media",
    "jigsaw_unintended_bias100K": "sentiment analysis",
    "mercari_price_suggestion100K": "retail",
    "melbourne_airbnb": "real estate",
    "yelp_review_polarity": "sentiment analysis",
    "yelp_reviews": "sentiment analysis",
    "amazon_review_polarity": "sentiment analysis",
    "amazon_reviews": "sentiment analysis",
    "agnews": "media",
    "california_house_price": "real estate",
    "ethos_binary": "sentiment analysis",
    "wine_reviews": "sentiment analysis",
    "yahoo_answers": "sentiment analysis",
    "google_qa_answer_type_reason_explanation": "topic classification",
    "google_qa_question_type_reason_explanation": "topic classification",
    "bnp_claims_management": "insurance",
    "creditcard_fraud": "fraud detection",
    "imbalanced_insurance": "insurance",
    "walmart_recruiting": "retail",
}

all_datasets_tabular = [
    "ames_housing",
    "naval",
    "forest_cover",
    "santander_customer_satisfaction",
    "adult_census_income",
    "sarcos",
    "amazon_employee_access_challenge",
    "numerai28pt6",
    "mercedes_benz_greener",
    "mushroom_edibility",
    "connect4",
    "allstate_claims_severity",
    "santander_customer_transaction",
    "otto_group_product",
    "protein",
    "ieee_fraud",
    "higgs",
    "titanic",
    "bnp_claims_management",
    "creditcard_fraud",
    "imbalanced_insurance",
    "walmart_recruiting",
]

all_datasets_text = [
    "sst2",
    "sst5",
    "bookprice_prediction",
    "fake_job_postings2",
    "jc_penney_products",
    "goemotions",
    "product_sentiment_machine_hack",
    "imdb_genre_prediction",
    "bbcnews",
    "jigsaw_unintended_bias100K",
    "mercari_price_suggestion100K",
    "melbourne_airbnb",
    "yelp_review_polarity",
    "yelp_reviews",
    "amazon_review_polarity",
    "amazon_reviews",
    "agnews",
    "california_house_price",
    "ethos_binary",
    "wine_reviews",
    "yahoo_answers",
    "reuters_r8",
    "ohsumed_7400",
    "google_qa_answer_type_reason_explanation",
    "google_qa_question_type_reason_explanation",
    "ae_price_prediction",
    "data_scientist_salary",
    "news_channel",
    "news_popularity2",
]

AMES_HOUSING_CONFIG = """
output_features:
  - name: SalePrice
    type: number
input_features:
  - name: MSSubClass
    type: category
  - name: MSZoning
    type: category
  - name: Street
    type: category
  - name: Neighborhood
    type: category
"""
MERCEDES_BENZ_GREENER_CONFIG = """
output_features:
  - name: y
    type: number
input_features:
  - name: X0
    type: category
  - name: X1
    type: category
  - name: X10
    type: binary
  - name: X11
    type: binary
  - name: X14
    type: binary
"""
BBCNEWS_CONFIG = """
output_features:
  - name: Category
    type: category
input_features:
  - name: Text
    type: text
"""

PRODUCT_SENTIMENT_MACHINE_HACK_NO_TEXT = """
input_features:
  - name: Product_Type
    type: category
    column: Product_Type
output_features:
  - name: Sentiment
    type: category
    column: Sentiment
"""

ADULT_CENSUS_INCOME = """
input_features:
  - name: age
    type: number
  - name: workclass
    type: category
  - name: fnlwgt
    type: number
  - name: education
    type: category
  - name: education-num
    type: number
  - name: marital-status
    type: category
  - name: occupation
    type: category
  - name: relationship
    type: category
  - name: race
    type: category
  - name: sex
    type: category
  - name: capital-gain
    type: number
  - name: capital-loss
    type: number
  - name: hours-per-week
    type: number
  - name: native-country
    type: category
output_features:
  - name: income
    type: binary
"""

FAKE_JOB_POSTINGS_MULTI_TO_TEXT = """
input_features:
  - name: description
    type: text
    column: description
  - name: required_experience
    type: category
    column: required_experience
  - name: required_education
    type: category
    column: required_education
output_features:
  - name: title
    type: text
    column: title
"""

TITANIC = """
input_features:
  - name: Pclass
    type: category
    column: Pclass
  - name: Sex
    type: category
    column: Sex
  - name: Age
    type: number
    column: Age
  - name: SibSp
    type: number
    column: SibSp
  - name: Parch
    type: number
    column: Parch
  - name: Fare
    type: number
    column: Fare
  - name: Embarked
    type: category
    column: Embarked
output_features:
  - name: Survived
    type: category
    column: Survived
"""

feature_type_to_config_for_encoder_preprocessing = {
    "number": (AMES_HOUSING_CONFIG, "ames_housing"),
    "category": (AMES_HOUSING_CONFIG, "ames_housing"),
    "binary": (MERCEDES_BENZ_GREENER_CONFIG, "mercedes_benz_greener"),
    "text": (BBCNEWS_CONFIG, "bbcnews"),
}


feature_type_to_config_for_decoder_loss = {
    "number": (AMES_HOUSING_CONFIG, "ames_housing"),
    "category": (PRODUCT_SENTIMENT_MACHINE_HACK_NO_TEXT, "product_sentiment_machine_hack"),
    "binary": (ADULT_CENSUS_INCOME, "adult_census_income"),
    "text": (FAKE_JOB_POSTINGS_MULTI_TO_TEXT, "fake_job_postings2"),
}
