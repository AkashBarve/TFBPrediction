
# coding: utf-8

# In[547]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import keras
import matplotlib.pyplot as plt


# In[548]:


train = pd.read_csv('train.csv')
train = train.sample(frac = 1)


# In[549]:


test = pd.read_csv('test.csv')


# In[550]:


dict = {'A':65,'C':67,'G':71,'T':84}


# In[551]:


def encode(data):
    Dlist = list(data)
    output = []
    for i in range (len(Dlist)):
        output.append(dict[Dlist[i]]/84)
    return np.array(output)


# In[552]:


train_seq = []
for i in range (len(train)):
    train_seq.append(encode(train.sequence[i]))
train_seq = np.array(train_seq)
label = np.array(train.label)


# In[553]:


val_idx = int(0.7 * len(train_seq))
x_train, x_val = train_seq[:val_idx], train_seq[val_idx:]
y_train, y_val = np.array(train.label)[:val_idx], np.array(train.label)[val_idx:]


# In[554]:


label


# In[555]:


num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


# In[556]:


model_1 = Sequential()
model_1.add(Dense(10, activation='relu', input_shape=(14,)))
model_1.add(Dropout(0.05))
#model_1.add(Dense(10, activation='relu'))
model_1.add(Dense(8, activation='relu'))
#model_1.add(Dense(8, activation='relu'))
#model_1.add(Dense(14, activation='relu'))
#model_1.add(Dense(14, activation='relu'))
#model_1.add(Dropout(0.4))
model_1.add(Dense(6, activation='relu'))
#model_1.add(Dense(6, activation='relu'))
model_1.add(Dense(4, activation='relu'))
#model_1.add(Dense(2, activation='relu'))
#model_1.add(Dense(7, activation='relu'))
#model_1.add(Dense(7, activation='relu'))
#model_1.add(Dropout(0.2))
model_1.add(Dense(2, activation='softmax'))


# In[557]:


model_1.summary()


# In[558]:


learning_rate = .001
model_1.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True),
              metrics=['accuracy'])


# In[559]:


batch_size = 32  # mini-batch with 128 examples
epochs = 120
history = model_1.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val))


# In[560]:


score = model_1.evaluate(x_val, y_val, verbose=0)[1]
score


# In[561]:


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    fig.savefig('Graph.png')

plot_loss_accuracy(history)


# In[562]:


test_seq = []
for i in range(len(test)):
    test_seq.append(encode(test.sequence[i]))
test_seq = np.array(test_seq)
result = model_1.predict(test_seq)

op = []
test_seq
submission = pd.DataFrame()
for i in range (len(result)):
    op.append(result[i].argmax())
op
submission['id'] = test['id']
submission['prediction'] = op
submission.to_csv('submission.csv', index=False)

