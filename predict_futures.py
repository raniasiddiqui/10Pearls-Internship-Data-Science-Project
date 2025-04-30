def predict_future(best_model, latest_features_df):
    return best_model.predict(latest_features_df.values)