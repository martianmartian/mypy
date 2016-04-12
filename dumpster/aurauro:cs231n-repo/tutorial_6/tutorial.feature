# Created by aurora at 16-2-24
Feature: data preprocessing Common pitfall.
  An important point to make about the preprocessing is that any preprocessing statistics
  (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data.
  E.g. computing the mean and subtracting it from every image across the entire dataset and then splitting the data into
  train/val/test splits would be a mistake. Instead, the mean must be computed only over the training data and then subtracted
  equally from all splits (train/val/test).

Feature: Weight Initialization
  That is, the recommended heuristic is to initialize each neuron's weight vector as: w = np.random.randn(n) / sqrt(n), where n is the number of its inputs. This ensures that all neurons in the network
  initially have approximately the same output distribution and empirically improves the rate of convergence.

Feature: Regularization
  It is most common to use a single, global L2 regularization strength that is cross-validated. It is also common to combine this with dropout applied after all layers. The value of p=0.5p=0.5 is a reasonable default,
  but this can be tuned on validation data.