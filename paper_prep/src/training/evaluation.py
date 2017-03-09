"""
class for evaluating models

"""




def accuracy(yhat, y):
    # print
    # print len(yhat)
    # print len(y)
    # print len(zip(yhat, y))
    return sum(1 if a == b else 0 for (a, b) in zip(yhat, y)) * 1.0 / len(y)



class Evaluator(object):
    def __init__(self, model, dataset):
        self.model_class = model_class
        self.data = dataset


    def nested_cross_validation(self, n_splits, hyperparam_options):
        hyperparams = []
        predictions = {}
        data_iterator = DataIterator(self.data)
        for test_set, test_indices, nested_set in data_iterator.xval_split(n_splits):
            best_hyperparams = self.cross_validate(n_splits - 1, hyperparam_options, nested_set)
            model = self.model_class(hyperparam_options)
            model.fit(test_set)

            hyperparams.append(best_hyperparams)

            predictions = model.predict(test_set)
            for i, p in zip(test_indices, predictions):
                predictions[i] = p

        # evaluate predictions, return hyperparams
        # TODO FROM HERE



    def cross_validate(self, n_splits, hyperparam_options, data):
        data_iterator = DataIterator(data)
        best_hyperparams = None
        for hyperparams in hyperparam_options:
            for val_set, val_indices, train_set in data_iterator.xval_split(n_splits):
                model.fit(train_set)
                preds = model.predict(val_set)

                # if not best_hyperparams or its somehow better, update
                # TODO FROM HERE

