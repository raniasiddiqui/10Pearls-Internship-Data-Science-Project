import joblib
def store_in_feature_store(features_df, target_list):
    import hopsworks
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS')
    fs = project.get_feature_store()

    print("Creating or getting feature group (version 2)...")
    fg = fs.get_or_create_feature_group(
        name="air_quality_features",
        version=2,  # NEW VERSION to accommodate updated schema
        primary_key=['hour', 'day', 'month'],
        description="Air quality features and targets v2"
    )

    print("Preparing data for insertion...")
    data = features_df.copy()
    data['target_aqi'] = target_list.astype('float32')  # Ensure compatible dtype

    print("Inserting data into feature group...")
    fg.insert(data)
    print("✅ Feature data stored successfully!")
    return fg
def store_model_in_registry(results, project):
    mr = project.get_model_registry()

    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = results[best_model_name]['model']
    best_metrics = {k: v for k, v in results[best_model_name].items() if k != 'model'}

    joblib.dump(best_model, 'best_model.pkl')
    model = mr.python.create_model(
        name="air_quality_predictor",
        metrics=best_metrics,
        description=f"Best model: {best_model_name}"
    )
    model.save('best_model.pkl')
    return best_model, best_model_name