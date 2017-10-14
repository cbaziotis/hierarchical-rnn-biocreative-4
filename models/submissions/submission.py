import datetime

import bioc
import numpy
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
from nltk import sent_tokenize
from sklearn.metrics import classification_report

from models.attention import Attention
from utils.data_helpers import load_data, vectorize_doc
from utils.load_embeddings import load_word_vectors


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


if __name__ == "__main__":
    # LOAD RAW DATA $ WORD VECTORS
    EVAL_DATASET = '../../dataset/PMtask_TestSet.xml'
    MODE = "eval"

    WV_PATH = '../../embeddings/PubMed-w2v.txt'
    WV_DIMS = 200
    MAX_SENT_LENGTH = 45
    MAX_SENTS = 23

    print("loading word embeddings...")
    word2idx, idx2word, embeddings = load_word_vectors(WV_PATH, WV_DIMS, True)

    docs, labels, ids = load_data(EVAL_DATASET, MODE)

    # convert strings to lists of tokens
    print("Tokenizing...")
    docs = [[text_to_word_sequence(sent) for sent in sent_tokenize(doc)]
            for doc in docs]

    # convert words to word indexes
    print("Vectorizing...")
    docs = [vectorize_doc(doc, word2idx, MAX_SENTS, MAX_SENT_LENGTH)
            for doc in docs]
    docs = numpy.array(docs)

    # LOAD SAVED MODEL
    print("Loading model from disk...", end=" ")
    model_name = "../experiments/task1_hGRU_2017-10-14 17:25:22.hdf5"
    filename = "../experiments/{}.hdf5".format(model_name)
    model = load_model(model_name, custom_objects={'Attention': Attention})
    print("done!")

    print(model.summary())
    pred = model.predict(docs)
    y_pred = np.around(pred)
    save_predictions(ids, relevant=y_pred, confidence=pred,
                     output='predictions.xml')
