# Module functions

----

## learning_curves


```python
ludwig.visualize.learning_curves(
  train_stats_per_model,
  output_feature_name,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show how model measures change over training and validation data epochs.

For each model and for each output feature and measure of the model,
it produces a line plot showing how that measure changed over the course
of the epochs of training on the training and validation sets.

__Inputs__


- __train_stats_per_model__ (list): List containing train statistics per model
- __output_feature_name__ (string): Name of the output feature that is predicted
   and for which is provided ground truth
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__

- __return__ (None):
 
----

## compare_performance


```python
ludwig.visualize.compare_performance(
  test_stats_per_model,
  output_feature_name,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparision barplot visualization for each overall metric


For each model (in the aligned lists of test_statistics and model_names)
it produces bars in a bar plot, one for each overall metric available
in the test_statistics file for the specified output_feature_name.

__Inputs__


- __test_stats_per_model__ (list): List containing train statistics per model
- __output_feature_name__ (string): Name of the output feature that is predicted and for which is provided ground truth
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_from_prob


```python
ludwig.visualize.compare_classifiers_performance_from_prob(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparision barplot visualization from probabilities.

For each model it produces bars in a bar plot, one for each overall metric
computed on the fly from the probabilities of predictions for the specified
output_feature_name.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __top_n_classes__ (list): List containing the number of classes to plot
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_from_pred


```python
ludwig.visualize.compare_classifiers_performance_from_pred(
  predictions_per_model,
  ground_truth,
  metadata,
  output_feature_name,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparision barplot visualization from predictions.

For each model it produces bars in a bar plot, one for each overall metric
computed on the fly from the predictions for the specified output_feature_name.

__Inputs__


- __predictions_per_model__ (list): List containing the model predictions
   for the specified output_feature_name
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __metadata__ (dict): Model's input metadata
- __output_feature_name__ (output_feature_name: output_feature_name containing ground trut):output_feature_name: output_feature_name containing ground truth
- __labels_limit__ (labels_limit: Maximum numbers of labels):labels_limit: Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (model_names: List of the names of the models to use as labels):model_names: List of the names of the models to use as labels.
- __output_directory__ (output_directory: Directory where to save plots):output_directory: Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (file_format: File format of output plots - pdf or p):file_format: File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_subset


```python
ludwig.visualize.compare_classifiers_performance_subset(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  subset,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparision barplot visualization from train subset.

For each model  it produces bars in a bar plot, one for each overall metric
computed on the fly from the probabilities predictions for the
specified output_feature_name, considering only a subset of the full training set.
The way the subset is obtained is using the top_n_classes and
subset parameters.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __top_n_classes__ (list): List containing the number of classes to plot
- __labels_limit__ (int): Maximum numbers of labels.
- __subset__ (): Type of the subset filtering
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
    If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_changing_k


```python
ludwig.visualize.compare_classifiers_performance_changing_k(
  probabilities_per_model,
  ground_truth,
  top_k,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produce lineplot that show Hits@K measure while k goes from 1 to top_k.


For each model it produces a line plot that shows the Hits@K measure
(that counts a prediction as correct if the model produces it among the
first k) while changing k from 1 to top_k for the specified output_feature_name.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __top_k__ (int): Number of elements in the ranklist to consider
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## compare_classifiers_multiclass_multimetric


```python
ludwig.visualize.compare_classifiers_multiclass_multimetric(
  test_stats_per_model,
  metadata,
  output_feature_name,
  top_n_classes,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show the precision, recall and F1 of the model for the specified output_feature_name.

For each model it produces four plots that show the precision,
recall and F1 of the model on several classes for the specified output_feature_name.

__Inputs__


- __test_stats_per_model__ (list): List containing train statistics per model
- __metadata__ (dict): Model's input metadata
- __output_feature_name__ (string): Name of the output feature that is predicted and for which is provided ground truth
- __top_n_classes__ (list): List containing the number of classes to plot
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__

- __return__ (None):
 
----

## compare_classifiers_predictions


```python
ludwig.visualize.compare_classifiers_predictions(
  predictions_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show two models comparision of their output_feature_name predictions.

__Inputs__


- __predictions_per_model__ (list): List containing the model predictions
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## confidence_thresholding_2thresholds_2d


```python
ludwig.visualize.confidence_thresholding_2thresholds_2d(
  probabilities_per_model,
  ground_truths,
  threshold_output_feature_names,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show confidence trethreshold data vs accuracy for two output_feature_name thresholds

The first plot shows several semi transparent lines. They summarize the
3d surfaces displayed by confidence_thresholding_2thresholds_3d that have
thresholds on the confidence of the predictions of the two
threshold_output_feature_names  as x and y axes and either the data coverage percentage or
the accuracy as z axis. Each line represents a slice of the data
coverage  surface projected onto the accuracy surface.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truths__ (list): List of NumPy Arrays containing ground truth data
- __threshold_output_feature_names__ (list): List of output_feature_names for 2d threshold
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (string): Name of the model to use as label.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## confidence_thresholding_2thresholds_3d


```python
ludwig.visualize.confidence_thresholding_2thresholds_3d(
  probabilities_per_model,
  ground_truths,
  threshold_output_feature_names,
  labels_limit,
  output_directory=None,
  file_format='pdf'
)
```


Show 3d confidence trethreshold data vs accuracy for two output_feature_name thresholds

The plot shows the 3d surfaces displayed by
confidence_thresholding_2thresholds_3d that have thresholds on the
confidence of the predictions of the two threshold_output_feature_names as x and y axes
and either the data coverage percentage or the accuracy as z axis.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truths__ (list): List of NumPy Arrays containing ground truth data
- __threshold_output_feature_names__ (list): List of output_feature_names for 2d threshold
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## confidence_thresholding


```python
ludwig.visualize.confidence_thresholding(
  probabilities_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models accuracy and data coverage while increasing treshold

For each model it produces a pair of lines indicating the accuracy of
the model and the data coverage while increasing a threshold (x axis) on
the probabilities of predictions for the specified output_feature_name.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (sting): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## confidence_thresholding_data_vs_acc


```python
ludwig.visualize.confidence_thresholding_data_vs_acc(
  probabilities_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models comparision of confidence treshold data vs accuracy.

For each model it produces a line indicating the accuracy of the model
and the data coverage while increasing a threshold on the probabilities
of predictions for the specified output_feature_name. The difference with
confidence_thresholding is that it uses two axes instead of three,
not visualizing the threshold and having coverage as x axis instead of
the threshold.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__

- __return__ (None):
 
----

## confidence_thresholding_data_vs_acc_subset


```python
ludwig.visualize.confidence_thresholding_data_vs_acc_subset(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  subset,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models comparision of confidence treshold data vs accuracy on a
subset of data.

For each model it produces a line indicating the accuracy of the model
and the data coverage while increasing a threshold on the probabilities
of predictions for the specified output_feature_name, considering only a subset of the
full training set. The way the subset is obtained is using the top_n_classes
and subset parameters.
The difference with confidence_thresholding is that it uses two axes
instead of three, not visualizing the threshold and having coverage as
x axis instead of the threshold.

If the values of subset is ground_truth, then only datapoints where the
ground truth class is within the top n most frequent ones will be
considered  as test set, and the percentage of datapoints that have been
kept  from the original set will be displayed. If the values of subset is
predictions, then only datapoints where the the model predicts a class
that is within the top n most frequent ones will be considered as test set,
and the percentage of datapoints that have been kept from the original set
will be displayed for each model.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __top_n_classes__ (list): List containing the number of classes to plot
- __labels_limit__ (int): Maximum numbers of labels.
- __subset__ (string): Type of the subset filtering
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## binary_threshold_vs_metric


```python
ludwig.visualize.binary_threshold_vs_metric(
  probabilities_per_model,
  ground_truth,
  metrics,
  positive_label=1,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show confidence of the model against metric for the specified output_feature_name.

For each metric specified in metrics (options are f1, precision, recall,
accuracy), this visualization produces a line chart plotting a threshold
on  the confidence of the model against the metric for the specified
output_feature_name.  If output_feature_name is a category feature, positive_label indicates which is
the class to be considered positive class and all the others will be
considered negative. It needs to be an integer, to figure out the
association between classes and integers check the ground_truth_metadata
JSON file.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (list): List of NumPy Arrays containing ground truth data
- __metrics__ (f1, precision, recall):metrics: metrics to dispay (f1, precision, recall,
            accuracy)
- __positive_label__ (string): Label of the positive class
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## roc_curves


```python
ludwig.visualize.roc_curves(
  probabilities_per_model,
  ground_truth,
  positive_label=1,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show the roc curves for the specified models output output_feature_name.

This visualization produces a line chart plotting the roc curves for the
specified output_feature_name. If output_feature_name is a category feature, positive_label indicates
which is the class to be considered positive class and all the others will
be considered negative. It needs to be an integer, to figure out the
association between classes and integers check the ground_truth_metadata
JSON file.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (list): List of NumPy Arrays containing ground truth data
- __positive_label__ (string): Label of the positive class
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## roc_curves_from_test_statistics


```python
ludwig.visualize.roc_curves_from_test_statistics(
  test_stats_per_model,
  output_feature_name,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show the roc curves for the specified models output binary output_feature_name.

This visualization uses the output_feature_name, test_statistics and model_names
parameters. output_feature_name needs to be binary feature. This visualization produces a
line chart plotting the roc curves for the specified output_feature_name.

__Inputs__


- __test_stats_per_model__ (list): List containing train statistics per model
- __output_feature_name__ (string): Name of the output feature that is predicted and for which is provided ground truth
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## calibration_1_vs_all


```python
ludwig.visualize.calibration_1_vs_all(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models probability of predictions for the specified output_feature_name.

For each class or each of the k most frequent classes if top_k is
specified,  it produces two plots computed on the fly from the
probabilities  of predictions for the specified output_feature_name.

The first plot is a calibration curve that shows the calibration of the
predictions considering the current class to be the true one and all
others  to be a false one, drawing one line for each model (in the
aligned  lists of probabilities and model_names).

The second plot shows the distributions of the predictions considering
the  current class to be the true one and all others to be a false one,
drawing the distribution for each model (in the aligned lists of
probabilities and model_names).

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __top_n_classes__ (list): List containing the number of classes to plot
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__String__


- __return__ (None):
 
----

## calibration_multiclass


```python
ludwig.visualize.calibration_multiclass(
  probabilities_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models probability of predictions for each class of the the
specified output_feature_name.

__Inputs__


- __probabilities_per_model__ (list): List of model probabilities
- __ground_truth__ (ndarray): NumPy Array containing ground truth data
- __labels_limit__ (int): Maximum numbers of labels.
     If labels in dataset are higher than this number, "rare" label
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## confusion_matrix


```python
ludwig.visualize.confusion_matrix(
  test_stats_per_model,
  metadata,
  output_feature_name,
  top_n_classes,
  normalize,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show confision matrix in the models predictions for each output_feature_name.

For each model (in the aligned lists of test_statistics and model_names)
it  produces a heatmap of the confusion matrix in the predictions for
each  output_feature_name that has a confusion matrix in test_statistics. The value of
top_n_classes limits the heatmap to the n most frequent classes.

__Inputs__


- __test_stats_per_model__ (string): List containing train statistics per model
- __metadata__ (dict): Model's input metadata
- __output_feature_name__ (string): Name of the output feature that is predicted and for which is provided ground truth
- __top_n_classes__ (list): List containing the number of classes to plot
- __normalize__ (bool): Flag to normalize rows in confusion matrix
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 
----

## frequency_vs_f1


```python
ludwig.visualize.frequency_vs_f1(
  test_stats_per_model,
  metadata,
  output_feature_name,
  top_n_classes,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show prediction statistics for the specified output_feature_name for each model.

For each model (in the aligned lists of test_statistics and model_names),
produces two plots statistics of predictions for the specified output_feature_name.

The first plot is a line plot with one x axis representing the different
classes and two vertical axes colored in orange and blue respectively.
The orange one is the frequency of the class and an orange line is plotted
to show the trend. The blue one is the F1 score for that class and a blue
line is plotted to show the trend. The classes on the x axis are sorted by
f1 score.
The second plot has the same structure of the first one,
but the axes are flipped and the classes on the x axis are sorted by
frequency.

__Inputs__


- __test_stats_per_model__ (list): List containing train statistics per model
- __metadata__ (dict): Model's input metadata
- __output_feature_name__ (string): Name of the output feature that is predicted and for which is provided ground truth
- __top_n_classes__ (list): List containing the number of classes to plot
- __model_names__ (list, default: None): List of the names of the models to use as labels.
- __output_directory__ (string, default: None): Directory where to save plots.
     If not specified, plots will be displayed in a window
- __file_format__ (string, default: 'pdf'): File format of output plots - pdf or png

__Return__


- __return__ (None):
 