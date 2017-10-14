from keras.layers import (Bidirectional, Dense, Embedding, GRU, Input,
                          TimeDistributed, GaussianNoise, Dropout, Lambda,
                          Concatenate)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from models.attention import Attention


def get_embedding_layer(embeddings, max_sent_length):
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=[embeddings],
        input_length=max_sent_length,
        mask_zero=True,
        trainable=False)
    return embedding_layer


def hrnn_title_abstract(embeddings, max_sents, max_sent_length, **kwargs):
    ######################################################
    # HyperParameters
    ######################################################
    drop_input = kwargs.get("drop_input", 0.2)
    noise_input = kwargs.get("noise_input", 0.2)
    rnn_size = kwargs.get("rnn_size", 150)

    rnn_rec_drop = kwargs.get("rnn_rec_drop", 0)
    rnn_drop = kwargs.get("rnn_drop", 0.3)
    att_drop = kwargs.get("att_drop", 0.3)

    #######################
    # WORDS RNN
    #######################
    # define input
    sentence_input = Input(shape=(max_sent_length,), dtype='int32')

    # embed words, using an Embedding layer
    embedding_layer = get_embedding_layer(embeddings, max_sent_length)
    words = embedding_layer(sentence_input)

    # Regularize embedding layer:
    # - add gaussian noise to word vectors
    words = GaussianNoise(noise_input)(words)
    # - add dropout to word vectors
    words = Dropout(drop_input)(words)

    # read each sentence, which is a sequence of words vectors
    # and generate a fixed vector representation.
    h_words = Bidirectional(GRU(rnn_size, return_sequences=True,
                                dropout=rnn_drop,
                                recurrent_dropout=rnn_rec_drop))(words)
    sentence = Attention()(h_words)
    sentence = Dropout(att_drop)(sentence)

    sent_encoder = Model(sentence_input, sentence)
    print(sent_encoder.summary())

    #######################
    # SENTENCE RNN
    #######################
    document_input = Input(shape=(max_sents, max_sent_length), dtype='int32')
    document_enc = TimeDistributed(sent_encoder)(document_input)

    # Now we have a single vector representation for each sentence.
    # Next just like before, we want to feed the vector of each sentence,
    # the sequence of sentence vectors, to another RNN, which will generate
    # a fixed vector representation for the whole document.
    h_sentences = Bidirectional(GRU(rnn_size, return_sequences=True,
                                    dropout=rnn_drop,
                                    recurrent_dropout=rnn_rec_drop))(
        document_enc)

    # extract the first sentence, which is the title
    title = Lambda(lambda x: x[:, 0, :])(h_sentences)

    # extract the rest of the sentences, which are the abstract
    # use attention, to get a single vector representation for the abstract
    abstract = Lambda(lambda x: x[:, 1:, :])(h_sentences)
    abstract = Attention()(abstract)
    abstract = Dropout(att_drop)(abstract)

    # concatenate the vector representations of the title and the abstract
    document = Concatenate()([title, abstract])

    #######################
    # CLASSIFIER
    #######################
    preds = Dense(1, activation='sigmoid', activity_regularizer=l2(.0001))(
        document)

    model = Model(document_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(clipnorm=5),
                  metrics=['acc'])

    return model


def hrnn_simple(embeddings, max_sents, max_sent_length, **kwargs):
    ######################################################
    # HyperParameters
    ######################################################
    drop_input = kwargs.get("drop_input", 0.2)
    noise_input = kwargs.get("noise_input", 0.2)
    rnn_size = kwargs.get("rnn_size", 150)

    rnn_rec_drop = kwargs.get("rnn_rec_drop", 0)
    rnn_drop = kwargs.get("rnn_drop", 0.2)
    att_drop = kwargs.get("att_drop", 0.2)

    #######################
    # WORDS RNN
    #######################
    # define input
    sentence_input = Input(shape=(max_sent_length,), dtype='int32')

    # embed words, using an Embedding layer
    embedding_layer = get_embedding_layer(embeddings, max_sent_length)
    words = embedding_layer(sentence_input)

    # Regularize embedding layer:
    # - add gaussian noise to word vectors
    words = GaussianNoise(noise_input)(words)
    # - add dropout to word vectors
    words = Dropout(drop_input)(words)

    # read each sentence, which is a sequence of words vectors
    # and generate a fixed vector representation.
    h_words = Bidirectional(GRU(rnn_size, return_sequences=True,
                                dropout=rnn_drop,
                                recurrent_dropout=rnn_rec_drop))(words)
    sentence = Attention()(h_words)
    sentence = Dropout(att_drop)(sentence)

    sent_encoder = Model(sentence_input, sentence)
    print(sent_encoder.summary())

    #######################
    # SENTENCE RNN
    #######################
    document_input = Input(shape=(max_sents, max_sent_length), dtype='int32')
    document_enc = TimeDistributed(sent_encoder)(document_input)

    # Now we have a single vector representation for each sentence.
    # Next just like before, we want to feed the vector of each sentence,
    # the sequence of sentence vectors, to another RNN, which will generate
    # a fixed vector representation for the whole document.
    h_sentences = Bidirectional(
        GRU(rnn_size, return_sequences=True,
            dropout=rnn_drop,
            recurrent_dropout=rnn_rec_drop)
    )(document_enc)

    document = Attention()(h_sentences)
    document = Dropout(att_drop)(document)
    #######################
    # CLASSIFIER
    #######################

    preds = Dense(1, activation='sigmoid', activity_regularizer=l2(.0001))(
        document)

    model = Model(document_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(clipnorm=5),
                  metrics=['acc'])

    return model
