from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import keras

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

num_samples = 1000
max_length = 1000
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answer = np.random.randint(0, answer_vocabulary_size, size=(num_samples, max_length))
answer = keras.utils.to_categorical(answer, answer_vocabulary_size)

text_input = Input(shape=(1000, ), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(1000, ), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encode_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encode_question], axis=-1)

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
model = Model([text_input, question_input], answer)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc']
)

model.fit([text, question], answer, epochs=10, batch_size=128)

# model.fit({'text': text, 'question': question}, answer, epochs=10, batch_size=128)


