from flask import Flask, request, render_template
import pickle
import tensorflow as tf
import json
import pandas as pd
import sqlite3
import time
from sklearn.base import BaseEstimator, TransformerMixin

def buildModel(optimizer, units=32):
    classifier = Sequential()
    classifier.add(Dense(units = units, activation = 'relu'))
    classifier.add(Dense(units = units, activation = 'relu'))
    classifier.add(Dense(units = 2, activation = 'softmax'))
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=[MulticlassAUC(pos_label=1, name="roc_auc")])
    return classifier

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def transform(self, X, y=None):
        return pd.to_datetime(X.Time, unit='s').dt.hour.to_frame()

class MulticlassAUC(tf.keras.metrics.AUC):
    """AUC for a single class in a muliticlass problem.

    Parameters
    ----------
    pos_label : int
        Label of the positive class (the one whose AUC is being computed).

    from_logits : bool, optional (default: False)
        If True, assume predictions are not standardized to be between 0 and 1.
        In this case, predictions will be squeezed into probabilities using the
        softmax function.

    sparse : bool, optional (default: True)
        If True, ground truth labels should be encoded as integer indices in the
        range [0, n_classes-1]. Otherwise, ground truth labels should be one-hot
        encoded indicator vectors (with a 1 in the true label position and 0
        elsewhere).

    **kwargs : keyword arguments
        Keyword arguments for tf.keras.metrics.AUC.__init__(). For example, the
        curve type (curve='ROC' or curve='PR').
    """

    def __init__(self, pos_label=1, from_logits=False, sparse=True, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse

    def update_state(self, y_true, y_pred, **kwargs):
        """Accumulates confusion matrix statistics.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values. Either an integer tensor of shape
            (n_examples,) (if sparse=True) or a one-hot tensor of shape
            (n_examples, n_classes) (if sparse=False).

        y_pred : tf.Tensor
            The predicted values, a tensor of shape (n_examples, n_classes).

        **kwargs : keyword arguments
            Extra keyword arguments for tf.keras.metrics.AUC.update_state
            (e.g., sample_weight).
        """
        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label]

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = y_pred[..., self.pos_label]

        super().update_state(y_true, y_pred, **kwargs)

def save_transaction(transaction):
    
    connection = sqlite3.connect('transactions.db')
    cursor = connection.cursor()
    text = f'''
    INSERT INTO trs (Time,{",".join([f"V{i}" for i in range(1, 29)])}, Amount, target,  'from', 'to', predict)
    VALUES ({",".join(["?" for _ in range(34)])})
    '''
    cursor.execute(text, list(transaction.values()))
    connection.commit()
    connection.close()

#загрузить
with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)
    

app = Flask(__name__)

@app.route('/handler', methods=['POST'])
def processing():

    transaction = json.loads(request.json)
    y_pred = model.predict(pd.DataFrame.from_dict([transaction]), verbose=0).item()
    transaction['predict'] = y_pred
    # transaction['Time'] = time.time()
    save_transaction(transaction)
    return "", 204

if __name__ == '__main__':
    app.run(port=5001)