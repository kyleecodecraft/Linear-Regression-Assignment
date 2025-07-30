import unittest
import main as tm

class TestTaxiModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = tm.load_data("data/chicago_taxi_train.csv")
        cls.df = tm.add_trip_minutes(cls.df)

    def test_load_data(self):
        self.assertGreater(len(self.df), 0, "Dataset should not be empty.")

    def test_model_training_single_feature(self):
        model, output = tm.run_experiment(self.df, ["TRIP_MILES"], "FARE", 0.001, 5, 10)
        weights, bias, epochs, rmse = output
        self.assertEqual(len(weights), 1)
        self.assertTrue(all(isinstance(x, float) or isinstance(x, np.float32) for x in rmse))

    def test_model_training_two_features(self):
        model, output = tm.run_experiment(self.df, ["TRIP_MILES", "TRIP_MINUTES"], "FARE", 0.001, 5, 10)
        weights, bias, epochs, rmse = output
        self.assertEqual(len(weights), 2)
    
    def test_add_trip_minutes(self):
        modified_df = tm.add_trip_minutes(self.df)
        self.assertIn("TRIP_MINUTES", modified_df.columns)
        self.assertTrue((modified_df["TRIP_MINUTES"] == modified_df["TRIP_SECONDS"] / 60).all())


    def test_prediction_output_shape(self):
        model, _ = tm.run_experiment(self.df, ["TRIP_MILES", "TRIP_MINUTES"], "FARE", 0.001, 5, 10)
        predictions = tm.predict_fare(model, self.df, ["TRIP_MILES", "TRIP_MINUTES"], "FARE", 10)
        self.assertEqual(predictions.shape[0], 10)
        self.assertIn("PREDICTED_FARE", predictions.columns)
        self.assertIn("OBSERVED_FARE", predictions.columns)
        self.assertIn("L1_LOSS", predictions.columns)

if __name__ == "__main__":
    unittest.main()