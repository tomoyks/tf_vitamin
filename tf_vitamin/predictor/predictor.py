class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, dataset):
        ds = dataset.map(lambda image, id_num: image)
        probabilities = self.model.predict(ds).flatten()
        return probabilities
