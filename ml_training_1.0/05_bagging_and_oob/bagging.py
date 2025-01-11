import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []

        for bag_indices in self.indices_list:
            model = model_constructor()
            data_bag = data[bag_indices] # Your Code Here
            target_bag = target[bag_indices]
            self.models_list.append(model.fit(data_bag, target_bag))

        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your Code Here
        predictions = np.array([model.predict(data) for model in self.models_list])
        return predictions.mean(axis=0)

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for model, bag_indices in zip(self.models_list, self.indices_list):
            oob_indices = np.setdiff1d(np.arange(len(self.data)), bag_indices)
            oob_data = self.data[oob_indices]
            oob_predictions = model.predict(oob_data)

            for idx, pred in zip(oob_indices, oob_predictions):
                list_of_predictions_lists[idx].append(pred)

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        # Your Code Here
        self.oob_predictions = np.array([
            np.mean(preds) if len(preds) > 0 else np.nan
            for preds in self.list_of_predictions_lists
        ], dtype=float)


    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        # Your Code Here
        valid_indices = ~np.isnan(self.oob_predictions)  # Проверяем, где есть действительные предсказания
        mse = np.mean((self.target[valid_indices] - self.oob_predictions[valid_indices]) ** 2)
        return mse
