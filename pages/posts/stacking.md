---
title: Model Stacking in Tensorflow
date: 2023/05/21
description: A guide to deterministic model stacking in Tensorflow
tag: ml, tensorflow, python
author: Marwan
---


# Brief

This article showcases how to perform model stacking in Tensorflow. 

Referencing the book [Designing Machine Learning Systems](https://www.amazon.com/Designing-Machine-Learning-Systems-Performance/dp/1491959900), model stacking is defined as follows:
"Stacking means that you train base learners from the training data then create a meta-learner that combines the outputs of the base learners to output final predictions."

The meta-learner can be a simple linear combination of the base learners (e.g. an average), a deterministic selection of the base learners, or it can be a more complex model that learns how to combine the base learners.

In our case, we will be building a simple stacked model that deterministically selects between two models based on a binary flag. More specifically, we take a simplified example of predicting flight delays. 

The premise being that the pattern in flight delays changes drastically between weekdays and weekends. Therefore, we train one model for weekdays and another for weekends. We then build a stacked model that delegates to the weekday and weekend models to generate predictions depending on whether the day is a weekday or not.

Note that while our example is purely contrived, it is common in other usecases: for instance, rideshare prices will fluctuate on weekdays versus weekends, and flight ticket prices rise during holiday seasons. Companies might have different models to deal with cyclic and seasonal drifts. A stacked model can be used to combine the predictions of these models. 

Here is a brief outline of the article:

- Synthesize data for a simplified flight delay prediction problem 
- Produce two very simple single layer models
- Train the models on the synthesized data
- Build an initial stacked model using python
- Build a stacked model using tensorflow
- Generate predictions using the stacked model 
- Acknowledge issues with the initial stacked model attempt and improve the implementation

# Data

```py
import pandas as pd
import numpy as np

nrows = 10_000

df_weekday = pd.DataFrame(
    {
        "is_weekday": np.ones(nrows),
        "x": np.random.triangular(0, 0, 1, nrows),
    }
)
df_weekday["delay"] = df_weekday["x"] * 1

df_weekend = pd.DataFrame(
    {
        "is_weekday": np.zeros(nrows),
        "x": np.random.triangular(0, 0, 1, nrows),
    }
)

df_weekend["delay"] = df_weekend["x"] * 10_000
```

We build two dataframes `df_weekday` and `df_weekend` that contain the following columns:
- `is_weekday`: a binary indicator of whether the day is a weekday or not
- `x`: a random number between 0 and 1. x in our case is our only covariate. Perhaps this an autoregressive term computed from previous weekday delays.
- `delay`: the target variable that we want to predict

We can see that the `delay` variable is generated differently for weekdays and weekends. This is to simulate the fact that the pattern in flight delays is different between weekdays and weekends. 

# Individual Models

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

input_= Input(shape=(1,), name="x", dtype=tf.float32)
output = Dense(1, activation="linear")(input_)

model_weekday = tf.keras.Model(
    inputs=[input_],
    outputs=[output],
)

model_weekday.compile(
    loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
)
model_weekday.fit(
    x=df_weekday["x"], y=df_weekday["delay"], epochs=500, verbose=False)
model_weekday.save("model_weekend.tf")


input_= Input(shape=(1,), name="x", dtype=tf.float32)
output = Dense(1, activation="linear")(input_)

model_weekend = tf.keras.Model(
    inputs=[input_],
    outputs=[output],
)

model_weekend.compile(
    loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
)
model_weekend.fit(
    x=df_weekend["x"], y=df_weekend["delay"], epochs=500, verbose=False
)
model_weekend.save("model_weekend.tf")
```

We build two very simple models. Each model has a single input layer and a single output layer. The input layer takes in the `x` covariate and the output layer predicts the `delay` variable. We train each model and save the models to disk for later use.

We verify that the models learned the correct parameters by inspecting the weights of the output layer.

```python
model_weekday.layers[-1].get_weights()
```
returns
```python
[array([[1.]], dtype=float32), array([2.2437239e-12], dtype=float32)]
```

```python
model_weekend_loaded.layers[-1].get_weights()
```
returns
```python
[array([[9999.999]], dtype=float32), array([-0.00172211], dtype=float32)]
```

Close enough to 1 and 10,000 respectively!

# Stacked Model


## Python-based stacked model
Before we continue to construct a stacked model using tensorflow, we will first build the stacked model as a python class. It is worth noting that to certain teams this will be far from optimal given:

- If the delivery interface between ML engineers and data scientists is tensorflow model files, then this approach will violate the interface. (i.e. The interface will expect a tensorflow model file and not a python class). 
- Given that stacking is done in python and not tensorflow, performance will be suboptimal as the model will not be able to take advantage of tensorflow's graph optimizations. 

However, we will proceed with this python-based approach as it is easier to understand and debug.

```python
class StackedModel:
    def __init__(self, model_weekday, model_weekend):
        self.model_weekday = model_weekday
        self.model_weekend = model_weekend
    
    def predict(self, df):
        df.reset_index(drop=False, inplace=True)
        
        df_weekday = df[df["is_weekday"] == 1].copy()
        df_weekend = df[df["is_weekday"] == 0].copy()
        
        df_weekday["delay"] = self.model_weekday.predict(df_weekday[["x"]])
        df_weekend["delay"] = self.model_weekend.predict(df_weekend[["x"]])
        
        df_stacked = pd.concat([df_weekday, df_weekend], axis=0).sort_index()
        df_stacked.set_index("index", inplace=True)
        return df_stacked["delay"]
```

The python stacked model class takes in the weekday and weekend models at initialization. It then uses the `is_weekday` flag to determine which model to use to generate predictions. The predictions are then concatenated and returned as a single dataframe. The index is used to ensure that the predictions are returned in the same order as the input.

We instantiate the stacked model class and generate predictions.

```python
stacked_model = StackedModel(model_weekday, model_weekend)
```
We prepare a sample of 10 rows to generate predictions on.

```python
df_sample = pd.concat([df_weekday, df_weekend]).sample(n=10)
df_sample
```

|    |   is_weekday |         x |    delay |
|---:|-------------:|----------:|---------:|
| 3259 |            1 | 0.053757  | 0.053757 |
| 5957 |            0 | 0.121251  | 1212.51  |
| 3912 |            0 | 0.523853  | 5238.53  |
|  450 |            1 | 0.063464  | 0.063464 |
| 2181 |            1 | 0.040998  | 0.040998 |
| 5909 |            0 | 0.212242  | 2122.42  |
| 6306 |            1 | 0.014279  | 0.014279 |
| 8435 |            0 | 0.027703  | 277.032  |
| 5478 |            1 | 0.207841  | 0.207841 |
| 8265 |            0 | 0.495394  | 4953.94  |

```python
stacked_model.predict(df_sample)
```

| index |      delay |
|---:|-----------:|
| 3259 |   0.053757 |
| 5957 |1212.513062 |
| 3912 |5238.531738 |
|  450 |   0.063464 |
| 2181 |   0.040998 |
| 5909 |2122.421631 |
| 6306 |   0.014279 |
| 8435 | 277.030060 |
| 5478 |   0.207841 |
| 8265 |4953.936523 |

## Tensorflow stacked model
We now proceed with our first attempt at building a stacked model in Tensorflow

```python

is_weekday_input = tf.keras.layers.Input(
    shape=(1,), name="is_weekday", dtype=tf.float32
)

x_tensor = tf.keras.layers.Input(shape=(1,), name="x", dtype=tf.float32)

models = [
    model_weekday_loaded,
    model_weekend_loaded,
]

conditions = [
    tf.math.equal(is_weekday_input, 1),
    tf.math.equal(is_weekday_input, 0),
]

inputs = [is_weekday_input, x_tensor]
outputs = []
for idx, (model, condition) in enumerate(zip(models, conditions)):
    # mask the input tensor(s) based on the condition
    input_masked = tf.boolean_mask(x_tensor, condition)
    # pass the masked input tensor(s) to the model
    output = model(input_masked)
    outputs.append(output)

# collect the outputs into a single tensor
stacked_output = tf.keras.layers.concatenate(outputs, axis=0)
stacked_model = tf.keras.models.Model(inputs=inputs, outputs=stacked_output)
```

We build a stacked model that takes in the `is_weekday` and `x` inputs. The input is then masked based on the `is_weekday` flag. The masked input is then passed to the weekday or weekend model depending on the `is_weekday` flag. The output of the weekday and weekend models are then concatenated and returned as a single tensor.


We can see that the model is built correctly by plotting the model.
```python
tf.keras.utils.plot_model(
    stacked_model, to_file="stacked_model_attempt1.png", show_shapes=True
)
```
![stacked_model_attempt1.png](/images/stacked_model_attempt1.png)


We proceed to generate predictions using the stacked model.

```python
df_sample = pd.concat([df_weekday, df_weekend]).sample(n=10|    )
df_sample
```

|    |   is_weekday |         x |    delay |
|---:|-------------:|----------:|---------:|
|  5498 |            1 | 0.196918  | 0.196918 |
|  7801 |            1 | 0.647037  | 0.647037 |
|  4325 |            0 | 0.205884  | 2058.84  |
|  1272 |            0 | 0.165437  | 1654.37  |
|  1378 |            1 | 0.88302   | 0.88302  |
|  1864 |            1 | 0.252031  | 0.252031 |
|  3392 |            1 | 0.543791  | 0.543791 |
|  4605 |            0 | 0.410707  | 4107.07  |
|  8829 |            0 | 0.657242  | 6572.42  |
|  734 |            0 | 0.393974  | 3939.74  |


```python
stacked_input = {
    "x": tf.convert_to_tensor(df_stacked["x"].to_numpy(), dtype=tf.float32),
    "is_weekday": tf.convert_to_tensor(df_stacked["is_weekday"].to_numpy(), dtype=tf.int32),
}
stacked_model.run_eagerly = False
out = stacked_model.call(stacked_input, training=False)
pd.DataFrame(out)
```

|    |         0 |
|---:|----------:|
|  0 | 0.196918  |
|  1 | 0.647037  |
|  2 | 0.88302   |
|  3 | 0.252031  |
|  4 | 0.543791  |
|  5 | 2058.837158  |
|  6 | 1654.371704  |
|  7 | 4107.065918  |
|  8 | 6572.419434  |
|  9 | 3939.736816  |

We have a problem with the output of the stacked model. The output is in a different order than the input. We need to fix this. Perhaps if we rely on an index to keep track of the order of the input, we can use it to reorder the output. Here is our first attempt where we explicitly pass an index input, apply the masking and concatenation and return it as an additional output.

```python
is_weekday_input = tf.keras.layers.Input(
    shape=(1,), name="is_weekday", dtype=tf.float32
)
index_tensor = tf.keras.layers.Input(shape=(1,), name="index", dtype=tf.int32)
x_tensor = tf.keras.layers.Input(shape=(1,), name="x", dtype=tf.float32)
models = [
    model_weekday_loaded,
    model_weekend_loaded,
]

conditions = [
    tf.math.equal(is_weekday_input, 1),
    tf.math.equal(is_weekday_input, 0),
]

inputs = [is_weekday_input, x_tensor, index_tensor]
outputs = []
index_masked = []
for idx, (model, condition) in enumerate(zip(models, conditions)):
    # mask the input tensor(s) based on the condition
    input_masked = tf.boolean_mask(x_tensor, condition)
    # mask the index tensor based on the condition
    index_masked.append(tf.boolean_mask(index_tensor, condition))
    # pass the masked input tensor(s) to the model
    output = model(input_masked)
    outputs.append(output)

index_after_mask = tf.keras.layers.concatenate(index_masked, axis=0)
stacked_output = tf.keras.layers.concatenate(outputs, axis=0)
stacked_model = tf.keras.models.Model(
    inputs=inputs, outputs=[stacked_output, index_after_mask]
)
```

We plot the model to verify that it is built correctly.

```python
tf.keras.utils.plot_model(
    stacked_model, to_file="stacked_model_attempt2.png", show_shapes=True
)
```
![stacked_model_attempt2.png](/images/stacked_model_attempt2.png)

We generate predictions using the stacked model.

```python
stacked_input = {
    "x": tf.convert_to_tensor(df_stacked["x"].to_numpy(), dtype=tf.float32),
    "index": tf.convert_to_tensor(
        df_stacked["index"].to_numpy(), dtype=tf.int32
    ),
    "is_weekday": tf.convert_to_tensor(
        df_stacked["is_weekday"].to_numpy(), dtype=tf.int32
    ),
}

stacked_model.run_eagerly = False
out = stacked_model.call(stacked_input, training=False)
delay, index = out
```

We inspect the predicted delay and the index.

```python
pd.DataFrame(delay)
```

As expected, the predicted delay is still in a different order than the input.


|    |         0 |
|---:|----------:|
|  0 | 0.196918  |
|  1 | 0.647037  |
|  2 | 0.88302   |
|  3 | 0.252031  |
|  4 | 0.543791  |
|  5 | 2058.837158  |
|  6 | 1654.371704  |
|  7 | 4107.065918  |
|  8 | 6572.419434  |
|  9 | 3939.736816  |

We now inspect the index.

```python
pd.DataFrame(index)
```

|    |   0 |
|---:|----:|
|  0 |   0 |
|  1 |   1 |
|  2 |   4 |
|  3 |   5 |
|  4 |   6 |
|  5 |   2 |
|  6 |   3 |
|  7 |   7 |
|  8 |   8 |
|  9 |   9 |


To re-order the predicted delay, we need to sort the predicted delay by the index. We can do this by using the `tf.gather` function.

```python
delay_reordered = tf.gather(delay, tf.argsort(index))

pd.DataFrame(delay_reordered)
```

|    |         0 |
|---:|----------:|
|  0 | 0.196918  |
|  1 | 0.647037  |
|  2 | 0.543791  |
|  3 | 2058.837158  |
|  4 | 1654.371704  |
|  5 | 0.88302   |
|  6 | 0.252031  |
|  7 | 4107.065918  |
|  8 | 6572.419434  |
|  9 | 3939.736816  |

We can see that the predicted delay is now in the same order as the input. 

We update the stacked model implementation to include the reordering of the predicted delay.

```python
is_weekday_input = tf.keras.layers.Input(
    shape=(1,), name="is_weekday", dtype=tf.float32
)
index_tensor = tf.keras.layers.Input(shape=(1,), name="index", dtype=tf.int32)
x_tensor = tf.keras.layers.Input(shape=(1,), name="x", dtype=tf.float32)

models = [
    model_weekday_loaded,
    model_weekend_loaded,
]

conditions = [
    tf.math.equal(is_weekday_input, 1),
    tf.math.equal(is_weekday_input, 0),
]

inputs = [is_weekday_input, index_tensor, x_tensor]
outputs = []
index_masked = []
for idx, (model, condition) in enumerate(zip(models, conditions)):
    # mask the input tensor(s) based on the condition
    input_masked = tf.boolean_mask(x_tensor, condition)
    # mask the index tensor based on the condition
    index_masked.append(tf.boolean_mask(index_tensor, condition))
    # pass the masked input tensor(s) to the model
    output = model(input_masked)
    outputs.append(output)

index_after_mask = tf.keras.layers.concatenate(index_masked, axis=0)
stacked_output = tf.keras.layers.concatenate(outputs, axis=0)
stacked_output_reordered = tf.gather(
    stacked_output, tf.argsort(index_after_mask)
)
stacked_model = tf.keras.models.Model(
    inputs=inputs, outputs=stacked_output_reordered
)
```

We plot the model to verify that it is built correctly.

```python
tf.keras.utils.plot_model(
    stacked_model, to_file="stacked_model_attempt3.png", show_shapes=True
)
```

![stacked_model_attempt3.png](/images/stacked_model_attempt3.png)

We generate predictions using the stacked model.

```python
stacked_input = {
    "x": tf.convert_to_tensor(df_stacked["x"].to_numpy(), dtype=tf.float32),
    "index": tf.convert_to_tensor(
        df_stacked["index"].to_numpy(), dtype=tf.int32
    ),
    "is_weekday": tf.convert_to_tensor(
        df_stacked["is_weekday"].to_numpy(), dtype=tf.int32
    ),
}

stacked_model.run_eagerly = False
out = stacked_model.call(stacked_input, training=False)
```

We inspect the predicted delay.

```python
pd.DataFrame(out)
```
|    |         0 |
|---:|----------:|
|  0 | 0.196918  |
|  1 | 0.647037  |
|  2 | 0.543791  |
|  3 | 2058.837158  |
|  4 | 1654.371704  |
|  5 | 0.88302   |
|  6 | 0.252031  |
|  7 | 4107.065918  |
|  8 | 6572.419434  |
|  9 | 3939.736816  |


We have now successfully re-ordered the predicted delay.


One final improvement. Notice that we don't have to explicitly pass an index to sort the inputs. Given the index is framed as a monotonic range of the same length as the input, we can build it dynamically from the input. Here is our second attempt where we dynamically build the index from the input.

Here is the updated stacked model implementation.

```python
is_weekday_input = tf.keras.layers.Input(
    shape=(1,), name="is_weekday", dtype=tf.float32
)

x_tensor = tf.keras.layers.Input(shape=(1,), name="x", dtype=tf.float32)

index_tensor = tf.keras.layers.Lambda(
    lambda x: tf.expand_dims(tf.range(tf.shape(x)[0]), axis=-1),
)(is_weekday_input)


models = [
    model_weekday_loaded,
    model_weekend_loaded,
]

conditions = [
    tf.math.equal(is_weekday_input, 1),
    tf.math.equal(is_weekday_input, 0),
]

inputs = [is_weekday_input, x_tensor]
outputs = []
index_masked = []
for idx, (model, condition) in enumerate(zip(models, conditions)):
    input_sliced = tf.boolean_mask(x_tensor, condition)
    index_masked.append(tf.boolean_mask(index_tensor, condition))
    output = model(input_sliced)
    outputs.append(output)

index_after_mask = tf.keras.layers.concatenate(index_masked, axis=0)
stacked_output = tf.keras.layers.concatenate(outputs, axis=0)
stacked_output_reordered = tf.gather(
    stacked_output, tf.argsort(index_after_mask)
)
stacked_model = tf.keras.models.Model(
    inputs=inputs, outputs=stacked_output_reordered
)
```

We plot the model to verify that it is built correctly.

```python
tf.keras.utils.plot_model(
    stacked_model, to_file="stacked_model_attempt4.png", show_shapes=True
)
```

![stacked_model_attempt4.png](/images/stacked_model_attempt4.png)

We generate predictions using the stacked model.

```python
stacked_input = {
    "x": tf.convert_to_tensor(df_stacked["x"].to_numpy(), dtype=tf.float32),
    "is_weekday": tf.convert_to_tensor(
        df_stacked["is_weekday"].to_numpy(), dtype=tf.int32
    ),
}

stacked_model.run_eagerly = False
out = stacked_model.call(stacked_input, training=False)
```

We inspect the predicted delay.

```python
pd.DataFrame(out)
```
|    |         0 |
|---:|----------:|
|  0 | 0.196918  |
|  1 | 0.647037  |
|  2 | 0.543791  |
|  3 | 2058.837158  |
|  4 | 1654.371704  |
|  5 | 0.88302   |
|  6 | 0.252031  |
|  7 | 4107.065918  |
|  8 | 6572.419434  |
|  9 | 3939.736816  |


The predicted delay is in the same order as the input. We have successfully built a stacked model in tensorflow.

# Conclusion

In this article, we have showcased how to build a stacked model in tensorflow. We have also highlighted some of the issues that we encountered along the way and how we resolved them.

# References

- [Designing Machine Learning Systems](https://www.amazon.com/Designing-Machine-Learning-Systems-Performance/dp/1491959900)


