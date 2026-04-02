import pickle

def make_predictions(model_path, data):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(data)
    return predictions

predictions = make_predictions("./model.pkl", [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])