import pickle
import random
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from IPython.display import SVG
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input, Flatten, Lambda, Bidirectional, concatenate, Dropout, GaussianNoise
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def l_f(file='Train.txt'):
    label = []
    cat_lab = []
    x_lab = []
    y_lab = []
    labels = []
    text = []
    lines = open(file, encoding="utf8").readlines()
    # random.shuffle(lines)
    # open('3mil.txt', 'w').writelines(lines)
    Length = len(lines)
    for i in range(len(lines)):
        label.append([])
        data = lines[i].split('\t\t')
        cat_lab.append(data[0])
        x_lab.append(data[1])
        y_lab.append(data[2])
        text.append(data[3].rstrip())

    return text, cat_lab, x_lab, y_lab


def load_file(file):
   '''This function is for opening and loading simple text of different type of question.
    '''
    text = []
    lines = open(file, encoding="utf8").readlines()
    random.shuffle(lines)
    for i in range(len(lines)):
        text.append(lines[i].rstrip())
    return text


def load_y(file):
    '''This function is for opening and loading y text of different type of question with only 1 inplace of each variable.
    '''
    y = []
    lines = open(file, encoding="utf8").read()
    spi = lines.split("', '")
    random.shuffle(spi)
    for i in range(len(spi)):
        y.append(spi[i])
    return y


def generator(v_size, tokenizer, max_l):

    ''' In this function we are loading different type of questions in text_Add.
    we are also loading the labels for the questions with 1 replacing each variable.
    Then we are taking random same number of random sample from each type and shuffle them
    convert them into X and y. This function will generate a random set of sample of the same type
    for fitting.  You can control the number of sample by setting up the value of counter.
    '''
    text_Add = load_file("Data/add.txt")
    text_Mul = load_file("Data/mul.txt")
    text_Sub = load_file("Data/sub.txt")
    text_Dev = load_file("Data/dev.txt")

    add_y = load_y("Data1/add_one")
    sub_y = load_y("Data1/sub_one")
    mul_y = load_y("Data1/mul_one")
    dev_y = load_y("Data1/dev_one")

    counter = 0
    count = 100
    batch = 5
    while True:
        X = []
        y = []
        result_add_X = random.sample((text_Add), count)
        result_add_Y = random.sample((add_y), count)
        result_sub_X = random.sample((text_Sub), count)
        result_sub_Y = random.sample((sub_y), count)
        result_mul_X = random.sample((text_Mul), count)
        result_mul_Y = random.sample((mul_y), count)
        result_dev_X = random.sample((text_Dev), count)
        result_dev_Y = random.sample((dev_y), count)
        for i in range(len(result_add_X)):
            X.append(result_add_X[i])
            y.append(result_add_Y[i])
        for i in range(len(result_sub_X)):
            X.append(result_sub_X[i])
            y.append(result_sub_Y[i])
        for i in range(len(result_mul_X)):
            X.append(result_mul_X[i])
            y.append(result_mul_Y[i])
        for i in range(len(result_dev_X)):
            X.append(result_dev_X[i])
            y.append(result_dev_Y[i])

        Shuf = list(zip(X, y))
        random.shuffle(Shuf)
        # Shuffling the X,y
        X, y = zip(*Shuf)
        trainX = tokenizer.texts_to_sequences(X)
        trainY = tokenizer.texts_to_sequences(y)

        train_padded_X = sequence.pad_sequences(trainX, maxlen=max_l, padding='post')
        train_padded_Y = sequence.pad_sequences(trainY, maxlen=max_l, padding='post')
        sample_y = to_categorical(train_padded_Y, v_size)
        counter = counter + 1
        if (counter >= batch):
            counter = 0
            yield (train_padded_X, sample_y)


def plot_words(data, start, stop, step, w_list):
    trace = go.Scatter(
        x=data[start:stop:step, 0],
        y=data[start:stop:step, 1],
        mode='markers',
        text=w_list[start:stop:step]
    )
    layout = dict(title='t-SNE 1 vs t-SNE 2',
                  yaxis=dict(title='t-SNE 2'),
                  xaxis=dict(title='t-SNE 1'),
                  hovermode='closest')
    fig = dict(data=[trace], layout=layout)
    py.offline.plot(fig, filename='model3.html')
    py.iplot(fig)


def main():
    # Storing the X as Data and Y as labels
    x, cat_lab, x_lab, y_lab = l_f()
    cat_LB = LabelBinarizer()
    cat_Elb = cat_LB.fit_transform(cat_lab)
    cat_Elb = np.expand_dims(cat_Elb, axis=1)
    text_labels = cat_LB.classes_
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

    with open('Test14.pickle', 'rb') as handle:
        token = pickle.load(handle)
    train1 = token.texts_to_sequences(X_train)
    test1 = token.texts_to_sequences(X_test)
    max_length = 70
    train_padded = sequence.pad_sequences(train1, maxlen=max_length)
    vocab_size = len(token.word_index)
    word_list = []
    for word, i in token.word_index.items():
        word_list.append(word)
    test_padded = sequence.pad_sequences(test1, maxlen=max_length)

    # # Building the Model
    num_epoch = 5

    # Main Model

    # Input Layer
    main_input = Input(shape=(max_length,), dtype='int32', name='main_input')
    # Embedding Layer
    x = Embedding(vocab_size, 100, input_length=max_length, name='emb')(main_input)
    # One Lstm Layer
    lstm_out = Bidirectional(LSTM(200, return_sequences=True, dropout=0.2))(x)

    def slicep(x):
        # return x[:, -1:, :]
        from keras.layers import concatenate
        return concatenate([x[:, :1, :], x[:, -1:, :]], axis=2)

    sliced = Lambda(slicep)(lstm_out)
    # Dense Layer
    prediction_output = Dense(len(text_labels), activation='sigmoid', name='prediction_Output')(sliced)
    variable_x_out = Dense(2, activation='softmax', name='variable_x_out')(lstm_out)
    variable_y_out = Dense(2, activation='softmax', name='variable_y_out')(lstm_out)
    # Text Generation Step
    add_x_y = concatenate([variable_x_out, variable_y_out], axis=2)

    def tile(y):
        import keras
        tiled = keras.backend.tile(y, (1, 70, 1))
        return tiled

    pre_tilling = Lambda(tile)(prediction_output)
    # add_pre_xy = concatenate([add_x_y, pre_tilling], axis=2)

    Drop_Out = Dropout(0.8)
    old_bi_lstm = Drop_Out(lstm_out, training=True)
    add_bilstm_pxy = concatenate([old_bi_lstm, pre_tilling, variable_x_out, variable_y_out])

    text_gen_lstm = LSTM(120, dropout=0.5, name="text_gen_LSTM", return_sequences=True)(add_bilstm_pxy)

    text = GaussianNoise(0.5)(text_gen_lstm, training=True)

    text_output = Dense(vocab_size, activation='softmax')(text)

    model = Model(inputs=[main_input],
                  outputs=[prediction_output, variable_x_out, variable_y_out],
                  name="MyModel")
    model2 = Model(
        inputs=[main_input],
        outputs=[text_output],
        name="MyModel2"
    )

    for layer in model.layers:
        layer.trainable = False

    print("Sumamry of Model 1&2", model.summary())
    print("Sumamry of Model 3", model2.summary())

    # Tensor Board
    tb = TensorBoard()
    plot_model(model, to_file='Model1.png')
    plot_model(model2, to_file='Model2.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))

    for layer in model.layers:
        layer.trainable = False

    model.compile(loss={'prediction_Output': 'categorical_crossentropy', 'variable_x_out': 'binary_crossentropy',
                        'variable_y_out': 'binary_crossentropy'},
                  sample_weight_mode="temporal",
                  optimizer='adam', metrics=['accuracy'])

    model.load_weights("w18Aug4_100.h5")
    # Visualization of the Embedding Layer

    # emb_layer = model2.get_layer(name='emb')
    # fn_layer = emb_layer.get_weights()[0]
    # lstm_tsne_embds = TSNE(n_components=2).fit_transform(fn_layer)
    # plot_words(lstm_tsne_embds, 0, 231, 1, word_list)

    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
    model2.fit_generator(generator(vocab_size, token, max_length), steps_per_epoch=300, epochs=num_epoch, verbose=1,
                         callbacks=[tb])

    model2.save("Model2.h5")
    model2.save_weights("wModel2.h5")
    print("Model is Saved")


if __name__ == '__main__':
    main()
