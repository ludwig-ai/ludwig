import json
import logging
import os
from typing import Dict

import torch

import openai
from ludwig.models.external import External

logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

evaluate_prompt_template = """
You a machine learning algorithm that can predict the best values for a new column
called `{{output_feature}}` based on the user provided data.

You infer the value of the new column value based on the values of the other columns.

You can use your current knowledge of the world to replace the training that the machine learning
algorithm would have received with.

The values for `{{output_feature}}` should be either `true` or `false` and nothing else.
If you do not know the answer, set `{{output_feature}}` to `false`.

Here are the columns the user will provide:
```
{{input_features}}
```

You always return your answer in JSON and only JSON with
the following keys: `{{output_feature}}`, `logit`, `probability`.

Here is an example of what you should return:
```
{"{{output_feature}}": true, "logits": 0.1, "probability": 0.9}
```

Here is the user input:
"""


class BaseGPT(External):
    def evaluate(self, inputs: Dict) -> Dict:
        """Evaluates inputs using the model."""
        # output_features max_length is 1
        output_feature = inputs["output_features"][0]
        input_features = ", ".join(inputs["input_features"])
        prompt = evaluate_prompt_template.replace("{{output_feature}}", output_feature).replace(
            "{{input_features}}", input_features
        )

        predictions = []
        logits = []
        probabilities = []
        for values in inputs["values"]:
            values = ", ".join([str(v) for v in values])
            content = f""""
            {input_features}
            {values}
            """
            logger.info(f"calling {self.model_name} with prompt: {prompt} and content: {content}")
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
            )
            content = completion.choices[0].message.content

            try:
                result = json.loads(content)
            except json.decoder.JSONDecodeError:
                logger.error(f"error decoding response into JSON: {content}")
                result = {
                    output_feature: False,
                    "logits": 0.0,
                    "probability": 0.0,
                }

            predictions.append(result[output_feature])
            logits.append(result["logits"])
            probabilities.append(result["probability"])

        return {
            output_feature: {
                # TODO (Andres) support other types of predictions
                "predictions": torch.Tensor(predictions).bool(),
                "logits": torch.Tensor(logits),
                "probabilities": torch.Tensor(probabilities),
            }
        }


class GPT4(BaseGPT):
    def __init__(self, random_seed: int = None):
        self.model_name = "gpt-4"
        super().__init__(random_seed)
