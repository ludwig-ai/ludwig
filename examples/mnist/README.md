# MNIST Hand-written Digit Classification

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
|create_training_test_data.sh|Bash script to create MNIST training and test data sets.|
|simple_model_training.py|Demonstrates using Ludwig api for training a model.|
|advance_model_training.py|Demonstrates a method to assess alternative model architectures.|
|assess_model_performance.py|Assess model performance on hold-out test data set.  This shows how to load a previously trained model to make predictions.|
|visualize_model_test_results.ipynb|Example for extracting training statistics and generate custom visualizations.|

