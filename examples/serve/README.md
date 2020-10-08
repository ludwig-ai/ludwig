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
retrieved 418 records for predictions
single record for prediciton:
 {'PassengerId': 892, 'Pclass': 3, 'Name': 'Kelly, Mr. James', 'Sex': 'male', 'Age': 34.5, 'SibSp': 0, 'Parch': 0, 'Ticket': '330911', 'Fare': 7.8292, 'Cabin': nan, 'Embarked': 'Q'}

invoking REST API /predict for single record...

Received 1 predictions
Sample predictions:
   Survived_predictions  Survived_probabilities
0                 False                0.257691

invoking REST API /batch_predict for entire dataframe...

Received 418 predictions
Sample predictions:
   Survived_predictions  Survived_probabilities
0                 False                0.257691
1                 False                0.439749
2                 False                0.439286
3                 False                0.146570
4                 False                0.427077
```
