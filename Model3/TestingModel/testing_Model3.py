from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# load the model
model = load_model('w19Aug.h5')

# load the tokenizer
tokenizer = load(open('Test14.pickle', 'rb'))

# select a seed text seed_text.append("Joan s cat had 8 kittens .
# She gave 2 to her friends . How many kittens does she have now ?") ,
seed_text = []
# What number do we get when we multiply 9 by 22 ?
#  joan s cat had 8 kittens . She gave 2 to her friends . How many kittens does she have now ?
seed_text.append('joan ')
encoded1 = tokenizer.texts_to_sequences(seed_text)
encoded = pad_sequences(encoded1, maxlen=70, padding='post')
preds = model.predict(encoded)[0]
prod = []
for i in range(70):
    sen = np.argmax(preds[i])
    prod.append(sen)
main_Pro = []
for i in prod:
    if (i > 0):
        main_Pro.append(i)
index_word = {v: k for k, v in tokenizer.word_index.items()}  # map back
words = []
for i in main_Pro:
    words.append(index_word.get(i))
    words.append(' ')
print(''.join(words))
