import numpy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import plot_model
from kutilities.callbacks import MetricsCallback, PlottingCallback
from nltk import sent_tokenize
from sklearn.metrics import f1_score

from models.nn_models import hrnn_title_abstract
from utils.data_helpers import load_data, data_splits, cp_name, vectorize_doc
from utils.load_embeddings import load_word_vectors

MODE = 'test'
CORPUS = 'dataset/PMtask_Triage_TrainingSet.xml'
WV_PATH = 'embeddings/PubMed-w2v.txt'
WV_DIMS = 200
####################
MAX_SENT_LENGTH = 45
MAX_SENTS = 23

##############################################
# Prepare Data
##############################################
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(WV_PATH, WV_DIMS, True)

print("loading data...")
docs, labels, ids = load_data(CORPUS, mode=MODE)
# word_vectors = load_word_vectors(args.embeddings)


# convert strings to lists of tokens
print("Tokenizing...")
docs = [[text_to_word_sequence(sent) for sent in sent_tokenize(doc)]
        for doc in docs]

# convert words to word indexes
print("Vectorizing...")
docs = [vectorize_doc(doc, word2idx, MAX_SENTS, MAX_SENT_LENGTH)
        for doc in docs]
docs = numpy.array(docs)

if MODE == "train":
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits(docs, labels,
                                                                 MODE)
elif MODE == "test":
    X_train, X_val, _, y_train, y_val, _ = data_splits(docs, labels, MODE)

##############################################
# Define Metrics and Callbacks
##############################################
metrics = {
    "f1_b": (lambda y_test, y_pred: f1_score(y_test, y_pred, average='binary')),
    "f1_m": (lambda y_test, y_pred: f1_score(y_test, y_pred, average='micro')),
    "f1_M": (lambda y_test, y_pred: f1_score(y_test, y_pred, average='macro')),
}

model_name = cp_name()
_datasets = {}
_datasets["1-train"] = (X_train, y_train)
_datasets["2-val"] = (X_val, y_val)
if MODE == "train":
    _datasets["3-test"] = (X_test, y_test)
metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics,
                                   batch_size=256)
plotting = PlottingCallback(grid_ranges=(0.3, 0.9), height=5,
                            plot_name=model_name)

# value to monitor
monitor_value = '2-val.f1_b'
# save the model whenever monitor_value increases (mode="max")
checkpointer = ModelCheckpoint(filepath=model_name,
                               monitor=monitor_value, mode="max",
                               verbose=1, save_best_only=True)
# stop the training if the monitor_value stops increasing (mode="max")
early_stop = EarlyStopping(monitor=monitor_value,
                           min_delta=0,
                           patience=6, verbose=1, mode='max')

_callbacks = []
_callbacks.append(metrics_callback)
_callbacks.append(plotting)
_callbacks.append(checkpointer)
_callbacks.append(early_stop)

##############################################
# Define Model and train it
##############################################

model = hrnn_title_abstract(embeddings, MAX_SENTS, MAX_SENT_LENGTH)
# model = hrnn_simple(embeddings, MAX_SENTS, MAX_SENT_LENGTH)
print(model.summary())
plot_model(model, to_file='model.png')

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=64,
          callbacks=_callbacks)
