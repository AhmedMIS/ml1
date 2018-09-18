
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing import sequence

# load our saved model
model = load_model('classification_Model_30epochs.h5')

# load tokenizer
tokenizer = Tokenizer()
with open('Test.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def load_file(file='Test'):
    lab = []
    text = []
    lines = open(file, encoding="utf8").readlines()
    for line in lines:
        data = line.split('\t\t')
        # print(data)
        if len(data) == 4:
            lab.append(data[0])
            text.append(data[3].rstrip())
    return text


# Storing the X as Data and Y as labels
x = load_file()

testing = load_file()
testing1 = tokenizer.texts_to_sequences(testing)
labels = ['add', 'dev', 'mul', 'sub']
max_len = 70

test_padded = sequence.pad_sequences(testing1, maxlen=max_len)
for i in test_padded:
    (pre, x, y) = model.predict(np.array([i]))
    label = labels[np.argmax(pre[0])]
    print("Type of question", label)
    x1 = np.argmax(x[0][:, 0])
    x2 = x[0][:, 0]
    y1 = np.argmax(y[0][:, 0])
    print("X and Y ", x1, y1)
