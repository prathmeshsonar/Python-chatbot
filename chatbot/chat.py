import random
import json

import torch
#import speech_recognition as s
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pyttsx3 as pp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


model.load_state_dict(model_state)
model.eval()


engine=pp.init()
voices=engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def speak(word):
    engine.say(word)
    engine.runAndWait()



bot_name = "Emily"
print("Let's chat! (type 'quit' to exit)")
while True:
   
    sentence =input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                op=(f"{random.choice(intent['responses'])}")
                o=(f"{bot_name}: " + op)
                print(o)
                speak(op)
    else:
        el=("I do not understand... For more information please contact : chatbot.support@protonmail.com")
        e=(f"{bot_name}: " + el)
        print(e)
        speak(el)
        
