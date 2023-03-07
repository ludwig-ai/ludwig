from explore_schema import (
    combine_configs,
    combine_configs_for_comparator_combiner,
    combine_configs_for_sequence_combiner,
)

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
  - name: SibSp
    type: number
    column: SibSp
  - name: Parch
    type: number
    column: Parch
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

ecd_config_section_to_config = {
    "trainer": (TITANIC, "titanic"),
    "comparator": (TITANIC, "titanic"),
    "concat": (TITANIC, "titanic"),
    "project_aggregate": (TITANIC, "titanic"),
    "sequence": (BBCNEWS_CONFIG, "bbcnews"),
    "sequence_concat": (BBCNEWS_CONFIG, "bbcnews"),
    "tabnet": (TITANIC, "titanic"),
    "tabtransformer": (TITANIC, "titanic"),
    "transformer": (TITANIC, "titanic"),
}

combiner_type_to_combine_config_fn = {
    "comparator": combine_configs_for_comparator_combiner,
    "concat": combine_configs,
    "project_aggregate": combine_configs,
    "sequence": combine_configs_for_sequence_combiner,
    "sequence_concat": combine_configs_for_sequence_combiner,
    "tabnet": combine_configs,
    "tabtransformer": combine_configs,
    "transformer": combine_configs,
}
