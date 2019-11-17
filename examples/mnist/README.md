# MNIST Hand-written Digit Prediction

This API example is based on [Ludwig's MNIST Hand-written Digit image classification example](https://uber.github.io/ludwig/examples/#image-classification-mnist). 

### Preparatory Steps
To create data for training and testing run this command:
```
cd mnist
./create_training_test_data.sh
```

This will create the following directory structure
```
mnist/
    data/
        mnist_dataset_training.csv
        mnist_dataset_testing.csv
        mnist_png/
            testing/
                0/
                . . .
                9/
            training/
                0/
                . . .
                9/
```

### Examples
|File|Description|
|----|-----------|
|simple_model_training.py|Demonstrates using Ludwig api for training a model.|
|multiple_model_training.py|Trains two models and generates a visualization for results of training.|
|model_training_results.ipynb|Example for extracting training statistics and generate custom visualizations.|

