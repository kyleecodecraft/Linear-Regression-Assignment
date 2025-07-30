# Linear-Regression-Assignment
This assignment is part of the AI Explorers course at CodeCraft Works.

## Assignment Steps

1. Create a python file named main.py. This will be the only file you use for this assignment.
2. Import pandas, matplotlib.pyplot, and seaborn.
3. Define a function named load_data(file_path) that reads a CSV file and returns the loaded DataFrame using pd.read_csv(file_path).
4. Define a function named build_model(lr, num_features) that takes in lr=learing rate and num_features=number of features being passed.
   1. function sets model inputs and outputs and compiles the model
   2. function returns model that it compiled
5. Define a function named train_model(model, features, label, epochs, batch_size) that takes in the parameters named.
   1. Create a history for the model using "model.fit()" you need to pass it specific variables.
   2. return the trained weight, trained bias, epochs and root mean squared error of the history.
6. Define a function run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size).
   1. This function needs to call build_model and train_model and pass the functions their required parameters
   2. function needs to return model and model output.
7. Define a function build_batch(df, batch_size), function needs to return a batch. This is a helper function for the future predict_fare function.
8. Define a function add_trip_minutes(df) which takes in the datafile (df)
   1. convert trip seconds to trip minutes
   2. Create a new column in the datafile for trip minutes
   3. pass the trip minutes data into this new column
9. Define a function predict_fare(model, df, features, label, batch_size=50)
   1.  this function calls build_batch and saves the returned batch
   2.  create a dataframe and pass the predicted fare, observed fare, and l1 loss into the dataframe
   3.  return the dataframe
10. Use the provided visualization functions to visualize your models
11. call run experiment with a model: learning rate=0.001, epochs=20, batch_size=50, features=["TRIP_MILES"], label='FARE'.
12. call run experiment with a new model. Keep all values the same except features=["TRIP_MILES", "TRIP_MINUTES"].
    1.  Use your conversion function to get trip minutes
13. Call predict_fare with the second model.

### Downloading Modules for Importing

If you are having trouble importing the correct modules, make sure they are downloaded into your codespace. In your terminal, type the following:

pip install pandas matplotlib seaborn

### Provided Code

~~~python
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
~~~