import logging

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.datasets import amazon_employee_access_challenge

df = amazon_employee_access_challenge.load()

model = LudwigModel(config="config.yaml", logging_level=logging.INFO)

training_statistics, preprocessed_data, output_directory = model.train(
    df,
    skip_save_processed_input=True,
    skip_save_log=True,
    skip_save_progress=True,
    skip_save_training_description=True,
    skip_save_training_statistics=True,
)

# Predict on unlabeled test
model.config["preprocessing"] = {}
unlabeled_test = df[df.split == 2].reset_index(drop=True)
preds, _ = model.predict(unlabeled_test)

# Save predictions to csv
action = preds.ACTION_probabilities_True
submission = pd.merge(unlabeled_test.reset_index(drop=True).id.astype(int), action, left_index=True, right_index=True)
submission.rename(columns={"ACTION_probabilities_True": "Action", "id": "Id"}, inplace=True)
submission.to_csv("submission.csv", index=False)
