import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() #stamming to eliminate extra characters like the '?' in 'there?' or 's' in 'Whats'

import numpy as np
import tflearn
import tensorflow
import random
import json

#-------------------------------------preprocessing the Data---------------------------------------

with open('intents.json') as file:
    data = json.load(file)

words = [] # list of words in patterns
labels = [] # list of labels
docs_x = [] # list of all the different patterns
docs_y = [] # list of tags

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"] # we dont want ?'s
words = sorted(list(set(words))) # use set for no dublicates, list converts set back to list, sorted sorts the words

labels = sorted(labels)

training = [] # bag of words containing 0's and 1's
output = [] # also list of 0'2 and 1's

out_empty = [0 for _ in range(len(labels))]

#One hot encoding the words

for x, doc in enumerate(docs_x):
     bag = [] #bag of words that exist / not exist as 0's and 1's

     wrds = [stemmer.stem(w) for w in doc]

     for w in words:
         if w in wrds:
             bag.append(1) # put 1 to tell that the words exists
         else:
             bag.append(0) # word doesnt exist

     output_row = out_empty[:]
     output_row[labels.index(docs_y[x])] = 1 # look through labels list to see where the tag is in the list and set that value zo 1 in output_row

     training.append(bag)
     output.append(output_row)

# for tflearn transform lists into numpy arrays
training = np.array(training)
output = np.array(output)

#------------------------------Create model and train model---------------------------------

net = tflearn.input_data(shape=[None, len(training[0])]) # input data with the length of our training data array
net = tflearn.fully_connected(net, 8) # add fully_connected layer to network. 8 neurons for this hidden layer each connected to the input data
net = tflearn.fully_connected(net, 8) # second hidden layer
net = tflearn.fully_connected(net,len(output[0]),activation="softmax") #softmax gives probability for each neuron in the outputlayer
net = tflearn.regression(net)

model = tflearn.DNN(net)

#if a trained model exists load it else train a new model and save it
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

#------------------------------------------Predictions------------------------------------------

#turn string input from user into bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

#handle prediction to grab a appropriate response from the JSON file
def chat():
    print("Rede mit dem Cookie-Shop ChatBot (zum schließen schreibe stop)")
    while True:
        inp = input("Ich: ")
        if inp.lower() == "stop":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()