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