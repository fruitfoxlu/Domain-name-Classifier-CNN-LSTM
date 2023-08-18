import numpy as np
#from keras.utils import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, LSTM
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data (you should adjust the paths accordingly)
dga_domains = pd.read_csv('dga_project_dga_domain_list_clean.txt', delimiter='\t', header=None, names=['family', 'domain', 'start_date', 'end_date'])
benign_domains = pd.read_csv('dga_project_top-1m.csv', header=None, names=['rank', 'domain']).drop(columns=['rank'])
#char_mapping = pd.read_csv('simple_char_dict.txt',delimiter=' ', header=None, names=['word','index'])

# Mapping function for domains to vectors
#char_to_num = {word: index+1 for index, word in char_mapping['word'].items()}

char_to_num = { 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52, '0': 53, '1': 54, '2': 55, '3': 56, '4': 57, '5': 58, '6': 59, '7': 60, '8': 61, '9': 62, ',': 64, ';': 65, '.': 66, '!': 67, '?': 68, ':': 69, '\'': 70, '/': 72, '\\': 73, '|': 74, '_': 75, '@': 76, '#': 77, '$': 78, '%': 79, '^': 80, '&': 81, '*': 82, '~': 83, '`': 84, '+': 85, '-': 86, '=': 87, '<': 88, '>': 89, '(': 90, ')': 91, '[': 92, ']': 93, '{': 94, '}': 95 }

def domain_to_vector(domain):
    domain_parts = domain.split('.')
    return [char_to_num.get(character, 0) for domain in domain_parts for character in domain]

# Convert domains to vectors
dga_domains['vector'] = dga_domains['domain'].apply(domain_to_vector)
benign_domains['vector'] = benign_domains['domain'].apply(domain_to_vector)

# Label the data
dga_domains['label'] = 1
benign_domains['label'] = 0

# Combine and split the data
combined_data = pd.concat([dga_domains, benign_domains], axis=0)
train_data, test_data = train_test_split(combined_data, test_size=0.2, stratify=combined_data['label'], random_state=42)

# Pad sequences
#max_length = max(combined_data['vector'].apply(len))
max_length = 30
X_train = pad_sequences(train_data['vector'], maxlen=max_length, padding='post')
X_test = pad_sequences(test_data['vector'], maxlen=max_length, padding='post')
y_train = train_data['label'].values
y_test = test_data['label'].values

# CNN Model definition
vocab_size = len(char_to_num) + 1
embedding_dim = 32

model = Sequential()
model.add(Embedding(input_dim=len(char_to_num) + 1, output_dim=128, input_length=max_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy on test data: {accuracy * 100:.2f}%")

#plot the result
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()