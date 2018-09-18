from keras.layers import Dense, Input, Flatten, Lambda, Bidirectional
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Model
import numpy as np
import pickle
from plt2 import x_plot, y_plot, pre_plot


def load_file(file='Train.txt'):
    label = []
    cat_lab = []
    x_lab = []
    y_lab = []
    text = []
    lines = open(file, encoding="utf8").readlines()
    # random.shuffle(lines)
    # open('3mil.txt', 'w').writelines(lines)
    for i in range(len(lines)):
        label.append([])
        data = lines[i].split('\t\t')
        cat_lab.append(data[0])
        x_lab.append(data[1])
        y_lab.append(data[2])
        text.append(data[3].rstrip())

   return text, cat_lab, x_lab, y_lab


def main():
    # Storing the X as Data and Y as labels
    x, cat_lab, x_lab, y_lab = load_file()
    cat_LB = LabelBinarizer()
    cat_Elb = cat_LB.fit_transform(cat_lab)
    cat_Elb = np.expand_dims(cat_Elb, axis=1)
    text_labels = cat_LB.classes_
    print("Order of Lables for testing", text_labels)

    x_train_labels = np.asarray([np.zeros((70, 2)) for i in range(len(cat_lab))])
    y_train_labels = np.asarray([np.zeros((70, 2)) for i in range(len(cat_lab))])

    for i in range(len(cat_lab)):
        x_train_labels[i, int(x_lab[i]), 0] = 1
        y_train_labels[i, int(y_lab[i]), 0] = 1
        # if (cat_lab[i] == 'add' or cat_lab[i] == 'mul'):
        #     x_train_labels[i, int(y_lab[i]), 0] = 1
        #     y_train_labels[i, int(x_lab[i]), 0] = 1
    x_train_labels[:, :, 1] = 1 - x_train_labels[:, :, 0]
    y_train_labels[:, :, 1] = 1 - y_train_labels[:, :, 0]
    '''
    For tackling the weight imbalance in our training set we are using sample weight for balacing our 
    training set. Fitting these sample weights with training data.  
    '''
    x_sample_weight = np.ones((x_train_labels.shape[0], x_train_labels.shape[1]))
    y_sample_weight = np.ones((y_train_labels.shape[0], y_train_labels.shape[1]))

    cat_sample_weight = np.ones((cat_Elb.shape[0], cat_Elb.shape[1]))

    x_sample_weight[:, :][np.where(x_train_labels[:, :, 0] == 0)] = 1 / 70.0

    y_sample_weight[:, :][np.where(y_train_labels[:, :, 0] == 0)] = 1 / 70.0

    # Spliting the data
    X_train, X_test, cat_lab_train, cat_lab_test, x_lab_train, x_lab_test, y_lab_train, y_lab_test, cat_sample_weight_train, \
    cat_sample_weight_test, x_sample_weight_train, x_sample_weight_test, y_sample_weight_train, y_sample_weight_test = \
        train_test_split(x, cat_Elb, x_train_labels, y_train_labels, cat_sample_weight, x_sample_weight,
                         y_sample_weight,
                         test_size=0.20, random_state=42)

    # TOkenising the Data
    token = Tokenizer(num_words=vocab_size, lower=True, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~?')
    token.fit_on_texts(x)
    train1 = token.texts_to_sequences(X_train)
    test1 = token.texts_to_sequences(X_test)
    vocab_size = len(token.word_index)
    max_len = 70
    train_padded = sequence.pad_sequences(train1, maxlen=max_len, dtype='int32')
    

    test_padded = sequence.pad_sequences(test1, maxlen=max_len)
    # # Building the Model
    num_epoch = 2
    # Main Model

    # Input Layer
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    # Embedding Layer
    emb_layer = Embedding(vocab_size, 100, input_length=None)(main_input)
    # One Lstm Layer
    lstm_out = Bidirectional(LSTM(250, return_sequences=True, dropout=0.1))(emb_layer)

    # Flattening and Slicing the out put of the Lstm layer
    def slicep(x):
        # return x[:, -1:, :]
        from keras.layers import concatenate
        return concatenate([x[:, :1, :], x[:, -1:, :]], axis=2)

    sliced = Lambda(slicep)(lstm_out)

    # Dense Layer
    prediction_output = Dense(len(text_labels), activation='softmax', name='prediction_Output')(sliced)
    variable_x_out = Dense(2, activation='softmax', name='variable_x_out')(lstm_out)
    variable_y_out = Dense(2, activation='softmax', name='variable_y_out')(lstm_out)

    # Initializing the Model with inpout and output layers
    model = Model(inputs=[main_input], outputs=[prediction_output, variable_x_out, variable_y_out], name="MyModel")
    print("Summary of the model", model.summary())
    # Compilining Model
    model.compile(loss={'prediction_Output': 'categorical_crossentropy', 'variable_x_out': 'binary_crossentropy',
                        'variable_y_out': 'binary_crossentropy'},
                  sample_weight_mode="temporal",
                  optimizer='adam', metrics=['accuracy'])

    print("These are model matrices ", cat_lab_test)

    class_weight = [cat_sample_weight_train, x_sample_weight_train, y_sample_weight_train]
    history = model.fit([train_padded],
                        [cat_lab_train, x_lab_train, y_lab_train],
                        sample_weight=class_weight,
                        validation_split=0.20, epochs=num_epoch, batch_size=64,
                        )

    # Saving the Model
    model.save('classification_Model2.h5')
    # Save Tokenizer i.e. Vocabulary
    with open('Test.pickle', 'wb') as f:
        pickle.dump(token, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model Is Save")
    print("Keys in the model", history.history.keys())
    History  = history.history
    with open('Model_history', 'wb') as f:
        pickle.dump(History, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model history is saved")

    # Ploting Accuracy of X , Y and prediction
    x_plot(History)
    y_plot(History)
    pre_plot(History)

    scores = model.evaluate([test_padded], [cat_lab_test, x_lab_test, y_lab_test], verbose=1)

    print("Titles", model.metrics_names)
    print("Score of the model", scores)


if __name__ == '__main__':
    main()
