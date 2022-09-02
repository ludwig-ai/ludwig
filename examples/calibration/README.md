# Calibration Examples

Drawing on the methods in
On Calibration of Modern Neural Networks (Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger), Ludwig supports
temperature scaling for binary and category output features. Temperature scaling brings a model’s output probabilities
closer to the true likelihood while preserving the same accuracy and top k predictions.

To enable calibration, add calibration: True to any binary or category feature:

```
output_features:
 - name: Cover_Type
   type: category
   calibration: True
```

With calibration enabled, Ludwig will attempt to find a scale factor (temperature) which will bring the probabilities
closer to the true likelihoods using the validation set. This calibration phase is run after training is complete. If
no validation set is provided, the training set is used for calibration.

To visualize the effects of calibration in Ludwig, you can use Calibration Plots, which bin the data based on model
probability and plot the model probability (X) versus the observed rate (Y) for each bin.
