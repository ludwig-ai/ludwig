# Ludwig Model Serve Example

This example shows Ludwig's http model serving capability, which is able to load a pre-trained Ludwig model and respond to REST APIs for predictions.
A simple client program illustrates how to invoke the REST API to retrieve predictions for provided input features.  The two REST APIs covered by this example:

|REST API|Description|
|--------|-----------|
|`/predict`|Single record prediction|
|`/batch_predict`|Prediction for batch of records|


### Preparatory Steps

* Run the `simple_model_training.py` example in `examples/titanic`. This should result the following file structures:
``` 
examples/
    titantic/
        results/
            simple_experiment_simple_model/
                model/
                description.json
                training_statistics.json
```


### Run Model Server Example

* Open two terminal windows
* In first terminal window:
  * Ensure current working directory is `examples/serve`
  * Start ludwig model server with the `titanic` trained model.  The following command uses the default host address (`0.0.0.0`) and port number (`8000`).
```
ludwig serve --model_path ../titanic/results/simple_experiment_simple_model/model
```

Sample start up messages for ludwig model server
```
███████████████████████
█ █ █ █  ▜█ █ █ █ █   █
█ █ █ █ █ █ █ █ █ █ ███
█ █   █ █ █ █ █ █ █ ▌ █
█ █████ █ █ █ █ █ █ █ █
█     █  ▟█     █ █   █
███████████████████████
ludwig v0.3 - Serve

INFO:     Started server process [4429]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

```

* In the second terminal window:
  * Ensure current working director is `examples/serve`
  * Run the sample client program

``` 
python client_program.py
```

Output should look like this
``` 
retrieved 1309 records for predictions
single record for prediction:
 {'PassengerId': 1, 'Survived': 0.0, 'Pclass': 3, 'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': nan, 'Embarked': 'S', 'split': 0}

invoking REST API /predict for single record...

Received 1 predictions
Sample predictions:
   Survived_predictions  Survived_probabilities_False  Survived_probabilities_True  Survived_probability
0                 False                      0.906132                     0.093868              0.906132

invoking REST API /batch_predict for entire dataframe...

Received 1309 predictions
Sample predictions:
   Survived_predictions  Survived_probabilities_False  Survived_probabilities_True  Survived_probability
0                 False                      0.906132                     0.093868              0.906132
1                  True                      0.165714                     0.834286              0.834286
2                  True                      0.441169                     0.558831              0.558831
3                  True                      0.228311                     0.771689              0.771689
4                 False                      0.878072                     0.121928              0.878072```
