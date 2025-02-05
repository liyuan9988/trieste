# %% [markdown]
# # Deep Ensemble Training Comparison
# 
# This notebook compares training of deep ensemble models with different batch sizes. We'll use:
# - Hartmann 6 objective function
# - 1,000 training observations
# - Two batch sizes: 64 and 1024
# - Tensorboard for logging training metrics
# 
# The architecture parameters are fixed:
# - Ensemble size: 10
# - Hidden layers: 6  
# - Units per layer: 150
# - Activation: swish

# %%
import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from gpflow.keras import tf_keras
import io
import time

from trieste.objectives import Hartmann6
from trieste.space import Box
from trieste.data import Dataset
from trieste.models.keras import DeepEnsemble, build_keras_ensemble
from trieste.models.optimizer import KerasOptimizer

# silence TF warnings and info messages, only print errors
tf.get_logger().setLevel("ERROR")

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Generate Dataset
# 
# We'll generate 20,000 random observations from the Hartmann 6 function.

# %%
# Generate training data
num_points = 1000
search_space = Hartmann6.search_space
inputs = search_space.sample(num_points)
outputs = Hartmann6.objective(inputs)
data = Dataset(inputs, outputs)

# Generate test data for evaluation
num_test = 10000
test_inputs = search_space.sample(num_test)
test_outputs = Hartmann6.objective(test_inputs)
test_data = Dataset(test_inputs, test_outputs)

# generate validation data
num_val = 10000
val_inputs = search_space.sample(num_val)
val_outputs = Hartmann6.objective(val_inputs)
val_data = Dataset(val_inputs, val_outputs)

# %%
class DeepEnsembleTest(DeepEnsemble):
    def optimize_encoded(self, dataset: Dataset) -> tf_keras.callbacks.History:
        """
        Optimize the underlying Keras ensemble model with the specified ``dataset``.

        Optimization is performed by using the Keras `fit` method, rather than applying the
        optimizer and using the batches supplied with the optimizer wrapper. User can pass
        arguments to the `fit` method through ``minimize_args`` argument in the optimizer wrapper.
        These default to using 100 epochs, batch size 100, and verbose 0. See
        https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
        arguments.

        Note that optimization does not return the result, instead optimization results are
        stored in a history attribute of the model object.

        :param dataset: The data with which to optimize the model.
        """
        fit_args = dict(self.optimizer.fit_args)

        # Tell optimizer how many epochs have been used before: the optimizer will "continue"
        # optimization across multiple BO iterations rather than start fresh at each iteration.
        # This allows us to monitor training across iterations.

        if "epochs" in fit_args:
            fit_args["epochs"] = fit_args["epochs"] + self._absolute_epochs

        x, y = self.prepare_dataset(dataset)
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((x, y))
            .prefetch(tf.data.experimental.AUTOTUNE)
            .repeat()
            .shuffle(dataset.observations.shape[0])
            .batch(fit_args["batch_size"], drop_remainder=True)
        )
        fit_args["batch_size"] = None
        history = self.model.fit(
            train_dataset,
            **fit_args,
            initial_epoch=self._absolute_epochs,
        )
        if self._continuous_optimisation:
            self._absolute_epochs = self._absolute_epochs + len(history.history["loss"])

        # Reset lr in case there was an lr schedule: a schedule will have changed the learning
        # rate, so that the next time we call `optimize` the starting learning rate would be
        # different. Therefore, we make sure the learning rate is set back to its initial value.
        # However, this is not needed for `LearningRateSchedule` instances.
        if not isinstance(
            self.optimizer.optimizer.lr, tf_keras.optimizers.schedules.LearningRateSchedule
        ):
            self.optimizer.optimizer.lr.assign(self.original_lr)

        return history


# %% [markdown]
# ## Model Building Function
# 
# Define a function to build and train models with different batch sizes.

# %%
def build_and_train_model(data: Dataset, epochs: int, batch_size: int, kappa: int, log_dir: str, profile: bool = False) -> DeepEnsemble:
    # Model architecture parameters
    ensemble_size = 10
    num_hidden_layers = 6
    num_nodes = 150
    
    # Build the ensemble
    keras_ensemble = build_keras_ensemble(
        data, 
        ensemble_size, 
        num_hidden_layers, 
        num_nodes,
        activation="swish"
    )
    
    # Learning rate schedule
    target_learning_rate = 0.001
    decay_epochs = int(0.95 * epochs)
    warmup_epochs = int(0.05 * kappa * epochs)
    steps_per_epoch = math.ceil(num_points / batch_size)
    lr_schedule = tf_keras.optimizers.schedules.CosineDecay(
        0.0, 
        decay_steps=decay_epochs * steps_per_epoch,
        alpha=0.0,
        warmup_target=target_learning_rate * np.sqrt(kappa),
        warmup_steps=warmup_epochs * steps_per_epoch,
    )
    
    # Training parameters
    fit_args = {
        "batch_size": batch_size,
        "epochs": warmup_epochs + decay_epochs,
        "verbose": 2,
        "callbacks": [
            tf_keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq='epoch'
                # update_freq='batch',
            )
        ]
    }
    
    optimizer = KerasOptimizer(
        tf_keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=1 - kappa *(1 - 0.9),
            beta_2=1 - kappa *(1 - 0.999),
            epsilon=1e-07 / np.sqrt(kappa),
            amsgrad=False,
            clipnorm=1.0,
        ), 
        fit_args
    )
    
    model = DeepEnsemble(keras_ensemble, optimizer)
    
    if profile:
        tf.profiler.experimental.start(log_dir)
    model.optimize(data, val_data)
    if profile:
        tf.profiler.experimental.stop()

    return model

# %% [markdown]
# ## Train Models with Different Batch Sizes

# %%
# Create log directories for each model
log_dir_base = "logs/deep_ensemble_comparison/"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_small_batch = os.path.join(log_dir_base, f"{timestamp}_batch_128")
log_dir_large_batch = os.path.join(log_dir_base, f"{timestamp}_batch_1024")

print("Training model with batch size 128...")
start_time_small = time.perf_counter()
model_small_batch = build_and_train_model(data, 1500, 128, 1, log_dir_small_batch)
end_time_small = time.perf_counter()
training_time_small = end_time_small - start_time_small
print(f"Training with batch size 128 took {training_time_small:.2f} seconds")

print("\nTraining model with batch size 1024...")
start_time_large = time.perf_counter()
model_large_batch = build_and_train_model(data, 1500, 1024, 8, log_dir_large_batch)
end_time_large = time.perf_counter()
training_time_large = end_time_large - start_time_large
print(f"Training with batch size 1024 took {training_time_large:.2f} seconds")

# %% [markdown]
# ## Compare Predictions
# 
# Let's compare predictions from both models against ground truth values.

# %%
def plot_predictions(model, data, test_data, title):
    # Get predictions for training data
    train_mean, train_var = model.predict(data.query_points)
    train_std = tf.sqrt(train_var)
    
    # Get predictions for test data
    test_mean, test_var = model.predict(test_data.query_points)
    test_std = tf.sqrt(test_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training data plot
    ax1.errorbar(
        data.observations.numpy().flatten(),
        train_mean.numpy().flatten(),
        yerr=2*train_std.numpy().flatten(),
        fmt='none',
        alpha=0.1,
        color='blue'
    )
    ax1.scatter(
        data.observations.numpy().flatten(),
        train_mean.numpy().flatten(),
        alpha=0.5,
        color='blue',
        label='Training predictions'
    )
    ax1.plot(
        [data.observations.numpy().min(), data.observations.numpy().max()],
        [data.observations.numpy().min(), data.observations.numpy().max()],
        'r--',
        label='Perfect fit'
    )
    ax1.set_xlabel('True values')
    ax1.set_ylabel('Predicted values')
    ax1.set_title(f'{title} - Training Data')
    ax1.legend()
    
    # Test data plot
    ax2.errorbar(
        test_data.observations.numpy().flatten(),
        test_mean.numpy().flatten(),
        yerr=2*test_std.numpy().flatten(),
        fmt='none',
        alpha=0.1,
        color='green'
    )
    ax2.scatter(
        test_data.observations.numpy().flatten(),
        test_mean.numpy().flatten(),
        alpha=0.5,
        color='green',
        label='Test predictions'
    )
    ax2.plot(
        [test_data.observations.numpy().min(), test_data.observations.numpy().max()],
        [test_data.observations.numpy().min(), test_data.observations.numpy().max()],
        'r--',
        label='Perfect fit'
    )
    ax2.set_xlabel('True values')
    ax2.set_ylabel('Predicted values')
    ax2.set_title(f'{title} - Test Data')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def log_matplotlib_figure(fig, log_dir: str, tag: str, step: int = 0):
    """
    Convert a matplotlib figure to an image and log it as a summary in TensorBoard.
    """
    # 1. Save the figure to a PNG in memory
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # 2. Decode PNG buffer as a TF image (shape: (height, width, channels))
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # 3. Add a batch dimension: (1, height, width, channels)
    image = tf.expand_dims(image, axis=0)

    # 4. Write to TensorBoard
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        tf.summary.image(tag, image, step=step)
    writer.close()

# Plot predictions for both models
fig1 = plot_predictions(model_small_batch, data, test_data, "Batch Size 128")
fig2 = plot_predictions(model_large_batch, data, test_data, "Batch Size 1024")

# Show them in the notebook
plt.show()

# Now log them in TensorBoard
log_matplotlib_figure(fig1, log_dir_small_batch, "Predictions_Batch_128", step=0)
log_matplotlib_figure(fig2, log_dir_large_batch, "Predictions_Batch_1024", step=0)

print("Logged prediction figures to TensorBoard!")

# %% [markdown]
# ## Analyze Results
# 
# To view the training metrics and compare the models:
# 
# 1. Start TensorBoard by running:
#    ```python
#    %load_ext tensorboard
#    %tensorboard --logdir logs/deep_ensemble_comparison
#    ```
# 
# 2. Compare the training curves, particularly:
#    - Loss convergence
#    - Validation metrics
#    - Training speed
# 
# 3. Look at the prediction plots above to compare:
#    - Prediction accuracy
#    - Uncertainty estimates (error bars)
#    - Any systematic biases
# 
# The smaller batch size (128) typically allows for:
# - More frequent model updates
# - Potentially better exploration of the loss landscape
# - May take longer to train
# 
# The larger batch size (1024) typically provides:
# - More stable gradient estimates
# - Faster training (fewer updates needed)
# - May converge to slightly worse minima

# %% [markdown]
# ## LICENSE
# 
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE) 