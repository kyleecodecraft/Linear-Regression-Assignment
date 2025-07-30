import pandas as pd
import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.metrics import RootMeanSquaredError

def load_data(url):
    return pd.read_csv(url)

def build_model(lr, num_features):
    inputs = Input(shape=(num_features,))
    outputs = Dense(units=1)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=RMSprop(learning_rate=lr),
        loss="mean_squared_error",
        metrics=[RootMeanSquaredError()])
    return model

def make_plots(df, feature_names, label_name, model_output, sample_size=200):
   random_sample = df.sample(n=sample_size).copy()
   random_sample.reset_index()
   
   # Do you remember what these hyperparameters mean?
   weights, bias, epochs, rmse = model_output
   is_2d_plot = len(feature_names) == 1
   model_plot_type = "scatter" if is_2d_plot else "surface"
   fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Loss Curve", "Model Plot"),
        specs=[[{"type": "scatter"}, {"type": model_plot_type}]])
    
   plot_data(random_sample, feature_names, label_name, fig)
   plot_model(random_sample, feature_names, weights, bias, fig)
   plot_loss_curve(epochs, rmse, fig)
   fig.show()
   return

def plot_loss_curve(epochs, rmse, fig):
   curve = px.line(x=epochs, y=rmse)
   curve.update_traces(line_color='#ff0000', line_width=3)
   
   fig.append_trace(curve.data[0], row=1, col=1)
   fig.update_xaxes(title_text="Epoch", row=1, col=1)
   fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])
   return

def plot_data(df, features, label, fig):
    if len(features) == 1:
        scatter = px.scatter(df, x=features[0], y=label)
    else:
        scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)
    
    fig.append_trace(scatter.data[0], row=1, col=2)
    if len(features) == 1:
        fig.update_xaxes(title_text=features[0], row=1, col=2)
        fig.update_yaxes(title_text=label, row=1, col=2)
    else:
        fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))
    return

def plot_model(df, features, weights, bias, fig):
    df['FARE_PREDICTED'] = bias[0]

    for index, feature in enumerate(features):
        df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

    if len(features) == 1:
        model = px.line(df, x=features[0], y='FARE_PREDICTED')
        model.update_traces(line_color='#ff0000', line_width=3)
    else:
        z_name, y_name = "FARE_PREDICTED", features[1]
        z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
        y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
        x = []
        for i in range(len(y)):
            x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
        colorscale=light_yellow))

    fig.add_trace(model.data[0], row=1, col=2)  
    return


def model_info(feature_names, label_name, model_output):
    weights = model_output[0]
    bias = model_output[1]
    nl = "\n"
    header = "-" * 80
    banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header
    info = ""
    equation = label_name + " = "

    for index, feature in enumerate(feature_names):
        info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
        equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

    info = info + "Bias: {:.3f}\n".format(bias[0])
    equation = equation + "{:.3f}\n".format(bias[0])
    return banner + nl + info + nl + equation

def train_model(model, features, label, epochs, batch_size):
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, verbose=0)
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs_ran = history.epoch
    rmse = history.history["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs_ran, rmse

def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):
    num_features = len(feature_names)
    features = df[feature_names].values
    label = df[label_name].values
    model = build_model(learning_rate, num_features)
    output = train_model(model, features, label, epochs, batch_size)
    print('{}'.format(model_info(feature_names, label_name, model_output)))
    return model, output

def build_batch(df, batch_size):
    batch = df.sample(n=batch_size).copy()
    batch.set_index(np.arange(batch_size), inplace=True)
    return batch

def add_trip_minutes(df):
    if 'TRIP_SECONDS' not in df.columns:
        raise ValueError("DataFrame must contain 'TRIP_SECONDS' column.")
    
    df = df.copy()
    df['TRIP_MINUTES'] = df['TRIP_SECONDS'] / 60.0
    return df

def predict_fare(model, df, features, label, batch_size=50):
    batch = build_batch(df, batch_size)
    predicted_values = model.predict_on_batch(batch[features].values)
    results = {
        "PREDICTED_FARE": [],
        "OBSERVED_FARE": [],
        "L1_LOSS": []
    }

    for i in range(batch_size):
        predicted = predicted_values[i][0]
        observed = batch[label].iloc[i]
        results["PREDICTED_FARE"].append(predicted)
        results["OBSERVED_FARE"].append(observed)
        results["L1_LOSS"].append(abs(observed - predicted))

    return pd.DataFrame(results)