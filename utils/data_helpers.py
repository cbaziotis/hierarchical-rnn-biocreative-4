import os
import pickle
from datetime import datetime

import numpy
from sklearn.model_selection import train_test_split
import bioc
import nltk


def triage2doc(corpus, mode, stop_words=True):
    """
    Extract title and abstract from a BioC collection.

    :param corpus: BioC collection
    :param stop_words: True if we want to remove stop words
    :return: a list where each item is a document with it title and abstract
    """
    ids = []
    texts = []
    labels = []
    stopwords = []
    if stop_words:
        for line in open('stopwords.txt'):
            stopwords.append(line.strip())

    with bioc.iterparse(corpus) as parser:
        for document in parser:
            ids.append(document.id)
            texts.append(extract_text(document, stopwords))
            if mode != 'eval':
                relevant = document.infons['relevant']
                labels.append(0 if relevant == 'no' else 1)

    return texts, labels, ids


def extract_text(document, stopwords):
    title_tokens = nltk.word_tokenize(document.passages[0].text)
    title_words = [word for word in title_tokens if
                   word.lower() not in stopwords]
    title = ' '.join(title_words)
    try:
        abstract_tokens = nltk.word_tokenize(document.passages[1].text)
        abstract_words = [word for word in abstract_tokens if
                          word.lower() not in stopwords]
        abstract = ' '.join(abstract_words)
    except IndexError:
        abstract = ''

    return title + ' ' + abstract


def vectorize_sent(text, word2idx, max_length):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        text (): the wordlist
        word2idx (): dictionary of word to ids
        max_length ():

    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    text = text[:max_length]

    for i, token in enumerate(text):
        if token in word2idx:
            words[i] = word2idx[token]
        else:
            words[i] = word2idx["<unk>"]

    return words


def vectorize_doc(doc, word2idx, max_sents, max_length):
    # trim sentences after max_sents
    doc = doc[:max_sents]
    _doc = numpy.zeros((max_sents, max_length), dtype='int32')
    for i, sent in enumerate(doc):
        s = vectorize_sent(sent, word2idx, max_length)
        _doc[i] = s
    return _doc


def cp_name():
    """Checkpoint file name"""
    filename = "task1_hGRU_{}.hdf5".format(datetime.now()
                                           .strftime("%Y-%m-%d %H:%M:%S"))
    return os.path.join(os.path.dirname(__file__),
                        "..", "models", "experiments", filename)


def load_data(corpus, mode):
    path, file = os.path.split(corpus)
    if mode == 'eval':
        _cache_file = os.path.join(path, "_data_eval.p")
    else:
        _cache_file = os.path.join(path, "_data.p")
    print("loading data...", end=" ")
    if os.path.exists(_cache_file):
        print("loading cached file...", end=" ")
        texts, labels, ids = pickle.load(open(_cache_file, "rb"))
    else:
        print("caching...", end=" ")
        texts, labels, ids = triage2doc(corpus, mode, stop_words=False)
        pickle.dump((texts, labels, ids), open(_cache_file, 'wb'))
    print("done!")
    return texts, labels, ids


def data_splits(data, labels, mode):
    """

    mode: (str) if mode=='train', then it will return 3 splits (train,val,test)
                if mode=='test', then it will return 2 splits (train,val)
    """
    # labels = to_categorical(numpy.asarray(labels))
    # labels_cat = onehot_to_categories(labels)
    labels = numpy.array(labels)

    if mode == 'train':
        X_train, X_rest, y_train, y_rest = train_test_split(data, labels,
                                                            test_size=0.2,
                                                            stratify=labels,
                                                            random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest,
                                                        test_size=0.5,
                                                        stratify=y_rest,
                                                        random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    elif mode == 'test':
        X_train, X_rest, y_train, y_rest = train_test_split(data, labels,
                                                            test_size=0.05,
                                                            stratify=labels,
                                                            random_state=42)
        return X_train, X_rest, None, y_train, y_rest, None
