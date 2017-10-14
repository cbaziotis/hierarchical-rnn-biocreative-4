import datetime

import bioc
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

from models.attention import Attention
from utils.data_helpers import load_data


def predict_classes(pred):
    if pred.shape[-1] > 1:
        return pred.argmax(axis=-1)
    else:
        return (pred > 0.5).astype('int32')


def eval_model(model, data, preds):
    pred = model.predict(data)
    y_pred = predict_classes(pred).ravel()
    print(classification_report(preds, y_pred, target_names=['no', 'yes']))


def save_predictions(ids, relevant, confidence, output):
    collection = bioc.BioCCollection()
    collection.source = 'PubMed'
    now = datetime.datetime.now()
    collection.date = '{}{:02d}{:02d}'.format(now.year, now.month, now.day)
    collection.key = 'collection.key'
    for i, id in enumerate(ids):
        document = bioc.BioCDocument()
        document.id = id
        document.infons['relevant'] = 'no' if relevant[i] == 0 else 'yes'
        if relevant[i] == 1:
            document.infons['confidence'] = '{:.2f}'.format(confidence[i][0])
        else:
            document.infons['confidence'] = '{:.2f}'.format(
                1 - confidence[i][0])
        collection.add_document(document)

    bioc.dump(collection, open(output, 'w'), pretty_print=True)


# LOAD RAW DATA $ WORD VECTORS
TRAINING_DATASET = '../../../dataset/PMtask_Triage_TrainingSet.xml'
EVAL_DATASET = '../../../dataset/PMtask_TestSet.xml'
MODE = "eval"
if MODE == 'eval':
    texts, labels, ids = load_data(EVAL_DATASET, MODE)
else:
    texts, labels, ids = load_data(TRAINING_DATASET, MODE)
# word_vectors = load_word_vectors("PubMed-w2v")

# LOAD SAVED MODEL
print("Loading model from disk...", end=" ")
model_name = "../experiments/task1_hGRU_2017-09-27 19:44:48.hdf5"
filename = "../experiments/{}.hdf5".format(model_name)
model = load_model(model_name, custom_objects={'Attention': Attention})
print("done!")

print(model.summary())

# tokenize the data and split it to train,val,set
MAX_SENT_LENGTH = 45
MAX_SENTS = 23
MAX_NB_WORDS = 50000
word_vectors = "PubMed-w2v"
_embeddings, _data = load_train_data(texts, labels, word_vectors, MODE,
                                     MAX_SENTS, MAX_SENT_LENGTH, logger)

# embedding_dim, word_index = _emb
X_train, X_val, X_test, y_train, y_val, y_test = _data

if MODE == 'eval':
    pred = model.predict(X_test)
    y_pred = np.around(pred)
    save_predictions(ids, relevant=y_pred, confidence=pred,
                     output='predictions.xml')
else:
    # evaluate the data
    print("EVAL TRAIN SET:\n---------------\n")
    eval_model(model, X_train, y_train)

    print("EVAL VAL SET:\n---------------\n")
    eval_model(model, X_val, y_val)

    if MODE == 'train':
        print("EVAL TEST SET:\n---------------\n")
        eval_model(model, X_test, y_test)
