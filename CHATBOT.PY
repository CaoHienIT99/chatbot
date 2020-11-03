import nltk 
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
with open("intents.json") as file :
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except: 
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:   
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
                labels.append(intent["tag"])   
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    words = sorted(list(set(words)))

    labels = sorted(labels)

    training =[]
    output = []

    out_emty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds: 
                bag.append(1)
            else:  
                bag.append(0)
        output_row = out_emty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output ), f) 

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try: 
    model.load("model.tflearn")
except: 
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")
#Import the library
from tkinter import *
root = Tk()
root.title("Chat Bot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)
main_menu = Menu(root)
# Create the submenu 
file_menu = Menu(root)

# Add commands to submenu
file_menu.add_command(label="New..")
file_menu.add_command(label="Save As..")
file_menu.add_command(label="Exit")
main_menu.add_cascade(label="File", menu=file_menu)
#Add the rest of the menu options to the main menu
main_menu.add_command(label="Edit")
main_menu.add_command(label="Quit")
root.config(menu=main_menu)
chatWindow = Text(root, bd=1, bg="black",  width="50", height="8", font=("Arial", 23), foreground="#00ffff")
chatWindow.place(x=6,y=6, height=385, width=370)
messageWindow = Text(root, bd=0, bg="black",width="30", height="4", font=("Arial", 23), foreground="#00ffff")
messageWindow.place(x=128, y=400, height=88, width=260)
scrollbar = Scrollbar(root, command=chatWindow.yview, cursor="star")
scrollbar.place(x=375,y=5, height=385)
Button= Button(root, text="Send",  width="12", height=5,
                    bd=0, bg="#0080ff", activebackground="#00bfff",foreground='#ffffff',font=("Arial", 12))
Button.place(x=6, y=400, height=88)
root.mainloop()
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w ==se:
                bag[i] = 1
    return numpy.array(bag)
def chatbot():
    print("Start talking with the Autochatbot! (type with to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "Quit":
            break

        Results = model.predict([bag_of_words(inp,words)])
        Results_index = numpy.argmax(Results)
        tag = labels[Results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))
chatbot()