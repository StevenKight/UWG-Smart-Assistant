"""
Main code for my personal AI named Wolfie.

A large portion of this code was written at different times and on different days over
the course of the past few years beginning in 2019.

This code utilizes a trained model that was created by the training.py file.
It then utilizes the microphone on the device to listen for you to say something.
it then determines what tag, from its json file, it should associate with your input and
utilizes that tag to determine an appropiate response.

Pylint: 8.63 (August 11, 2022)
"""

import calendar
import os
import sys
import pickle
import random
import time
import json
from datetime import date, datetime

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from playsound import playsound
from gtts import gTTS
import speech_recognition as sr

sys.path.append('Tasks')
import bot_math
from UWG import events

__author__ = "Steven Kight"
__version__ = "2.0"
__pylint__ = "2.14.4"

# Sets the word lemmatizer
LEMMATIZER = WordNetLemmatizer()

# Loads pre trained model and parts
MODEL = load_model('Tasks/Models/chatbot_model.h5')

INTENTS = json.load(open('Tasks/Models/intents.json'))
WORDS = pickle.load(open('Tasks/Models/words.pkl', 'rb'))
CLASSES = pickle.load(open('Tasks/Models/classes.pkl', 'rb'))

LIST_EVENTS = json.load(open('Tasks/UWG/Events.json'))

# Get students files from UWG
CUR_DIR = os.getcwd()
PATH = os.path.join(CUR_DIR, 'Dataset')

# Folders must be named after person
LIST_OF_STUDENTS = []

# open file and read the content in a list
with open('Tasks/UWG/students.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        LIST_OF_STUDENTS.append(currentPlace)


def bag_of_words(s, words):
    """
    Transfers the input words to understandable numbers.

    :param s: The sentence to process.
    :param words: Pickled file of words from trained model.
    :return: A numpy array representation of the bag of words.
    """

    sentence_words = nltk.word_tokenize(s)
    sentence_words = [LEMMATIZER.lemmatize(word.lower()) for word in sentence_words]

    bag = [0 for _ in range(len(words))]

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence, model):
    """
    Probability of each tag in intents file it could match.

    :param sentence: The sentence to be processed.
    :param model: The neural network model.
    :return: A probability list of likelyhood of each tag.
    """

    p = bag_of_words(sentence, WORDS)
    res = model.predict(np.array([p]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": CLASSES[r[0]], "probability": str(r[1])})

    return return_list


def getresponse(ints, intents_json):
    """
    Defines which tag has the highest probability.

    :param ints: A sorted probability list of likelyhood of tags.
    :param intents_json: The json file storing data.
    :return: The most likely tag and the probability of that tag.
    """

    tag = ints[0]['intent']
    probability = ints[0]['probability']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = tag

            return result, probability


def chatbot_response(msg):
    """
    Uses the two above code functions to return a response.

    :param msg: The sentence to be processed.
    :return: The most likely tag and the probability of that tag.
    """
    ints = predict_class(msg, MODEL)
    res, prob = getresponse(ints, INTENTS)
    return res, prob


def listening():
    """
    Utilizes SpeechRecognition api and the default microphone to
    turn what the user is saying into text.

    :return: String of what the user said aloud.
    """

    while True:
        r = sr.Recognizer()

        with sr.Microphone() as source: # use the default microphone as the audio source
            r.adjust_for_ambient_noise(source)  # here
            print("Wolfie is listening.")
            audio = r.listen(source, timeout = 10)

        try:
            text = r.recognize_google(audio)
            break
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            continue
        except sr.RequestError as e:
            print(f"Could not get results from Google Speech Recognition service; {0}".format(e))
            continue

    return text


def chat(start, people):
    """
    Defining of the loop for getting input, predicting and saying answer
    until users says goodbye or quit.

    :param start: The first thing the user said for startup sequence.
    :param people: list of people in the initial image (used for personalized questions)
    """

    language = 'en'

    start_tag = chatbot_response(start)[0]

    if start_tag == 'greeting':
        # Possibly structure a specialized name introduction
        introduction = ""

        if any("Unknown" in element for element in people):
            introduction = random.choice(i['responses']).format("")
        else:
            for place in range(len(people)):
                # Write a function to get the names of people and use mr and mrs
                # using an information.json file

                if " " in people[place]:
                    first_name = people[place].split(" ")[0]
                else:
                    first_name = people[place]

                if place == 0:
                    names = first_name
                elif place > 0:
                    names += " and " + first_name

            list_of_intents = INTENTS['intents']
            for i in list_of_intents:
                if i['tag'] == "greeting":
                    introduction = random.choice(i['responses']).format(names)

            myobj = gTTS(text=introduction, lang=language, slow=False)
            myobj.save("Voice.mp3")
            playsound("Voice.mp3")

    while True:

        if start_tag == 'greeting':
            start_tag = ''
            start = ''
            inp = listening()
        else:
            if start != '':
                inp = start
                start = ''
            else:
                inp = listening()

        print("WAIT")

        if "quit" in inp.lower():
            break

        if inp != '':
            tag = chatbot_response(inp)[0]

        if tag == "goodbye":
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    response = random.choice(i['responses'])

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")
                    break

            break

        if tag == "time":
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    string_response = random.choice(i['responses'])

                    response = string_response + str(time.strftime("%I:%M %p", time.localtime()))

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")

        elif tag == "date":
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    string_response = random.choice(i['responses'])

                    response = string_response + str(datetime.today().strftime('%Y-%m-%d'))

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")

        elif tag == "addition":
            numbers = []
            for word in inp.split():
                if word.isdigit():
                    numbers.append(int(word))

            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    string_response = random.choice(i['responses'])

                    response = string_response + str(bot_math.addlist(numbers))

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")

        elif tag == "multiplication":
            numbers = []
            for word in inp.split():
                if word.isdigit():
                    numbers.append(int(word))

            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    string_response = random.choice(i['responses'])

                    response = string_response + str(bot_math.multiplylist(numbers))

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")

        elif tag == "subtraction":
            numbers = []
            for word in inp.split():
                if word.isdigit():
                    numbers.append(int(word))

            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    string_response = random.choice(i['responses'])

                    response = string_response + str(bot_math.subtractlist(numbers))

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")

        elif tag == "division":
            numbers = []
            for word in inp.split():
                if word.isdigit():
                    numbers.append(int(word))

            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    string_response = random.choice(i['responses'])

                    response = string_response + str(bot_math.dividelist(numbers))

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")

        elif tag == "spelling":
            list_of_words = inp.split()
            next_word = list_of_words[list_of_words.index("spell") + 1]

            def split(sent):
                return [char for char in sent]

            response = str(next_word) + "   " + str(split(next_word))

            myobj = gTTS(text=response, lang=language, slow=False)
            myobj.save("Voice.mp3")
            playsound("Voice.mp3")

        elif tag == "areofcircle":
            if "radius" in inp:
                numbers = []
                for word in inp.split():
                    if word.isdigit():
                        numbers.append(int(word))

                response = bot_math.areaofcircle(int(numbers[0]))

                myobj = gTTS(text=response, lang=language, slow=False)
                myobj.save("Voice.mp3")
                playsound("Voice.mp3")

            else:
                response = "The formula is pie times the radius squared"

                myobj = gTTS(text=response, lang=language, slow=False)
                myobj.save("Voice.mp3")
                playsound("Voice.mp3")

        elif tag == "circumference":
            if "radius" in inp:
                numbers = []
                for word in inp.split():
                    if word.isdigit():
                        numbers.append(int(word))

                response = bot_math.circumfrance(int(numbers[0]))

                myobj = gTTS(text=response, lang=language, slow=False)
                myobj.save("Voice.mp3")
                playsound("Voice.mp3")

            else:
                response = "The formula is 2 times pie times the radius"

                myobj = gTTS(text=response, lang=language, slow=False)
                myobj.save("Voice.mp3")
                playsound("Voice.mp3")

        elif tag == "time_next_class":
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:

                    for name in LIST_OF_STUDENTS:
                        if name in people:

                            try:
                                list_classes = json.load(open(PATH + "/" + name + "/Current.json"))
                            except:
                                string_response = "I do not have any classes for you"
                                continue

                            response = ""
                            string_response = "Your next class is at "

                            curr_date = date.today()
                            day = calendar.day_name[curr_date.weekday()]

                            time_checked = datetime.today().strftime("%I:%M %p")
                            time_checked = datetime.strptime(time_checked,'%I:%M %p')

                            list_of_classes = list_classes['Current Classes']

                            day_count = curr_date.weekday()
                            while day_count <= 7:
                                for j in list_of_classes:
                                    if j['Time'] != 'Online':
                                        class_time = datetime.strptime(j['Time'],'%I:%M %p')
                                        if (day in j['Days']) and (time_checked <= class_time):
                                            response = string_response + j['Time']
                                            break
                                if response == "":
                                    day_count += 1
                                    if day_count == 7:
                                        day_count = 0
                                    else:
                                        pass
                                    day = calendar.day_name[day_count]

                                    time_checked = datetime.strptime('1:00 AM','%I:%M %p')
                                    string_response = "No classes for the rest of today"
                                    string_response += ", your next class is on {} at ".format(day)
                                else:
                                    break

                        else:
                            response = "I do not have any classes for you"

                    if response != "":
                        myobj = gTTS(text=response, lang=language, slow=False)
                        myobj.save("Voice.mp3")
                        playsound("Voice.mp3")
                    else:
                        print("error")

        elif tag == "locate_next_class":
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:

                    for name in LIST_OF_STUDENTS:
                        if name in people:

                            try:
                                list_classes = json.load(open(PATH + "/" + name + "/Current.json"))
                            except:
                                string_response = "I do not have any classes for you"
                                continue

                            response = ""
                            string_response = "Your class is in the "

                            curr_date = date.today()
                            day = calendar.day_name[curr_date.weekday()]

                            time_checked = datetime.today().strftime("%I:%M %p")
                            time_checked = datetime.strptime(time_checked,'%I:%M %p')

                            list_of_classes = list_classes['Current Classes']

                            day_count = curr_date.weekday()
                            while day_count <= 7:
                                for j in list_of_classes:
                                    if j['Time'] != 'Online':
                                        class_time = datetime.strptime(j['Time'],'%I:%M %p')
                                        if (day in j['Days']) and (time_checked <= class_time):
                                            response = string_response + j['Location']
                                            break
                                if response == "":
                                    day_count += 1
                                    if day_count == 7:
                                        day_count = 0
                                    else:
                                        pass
                                    day = calendar.day_name[day_count]

                                    time_checked = datetime.strptime('1:00 AM','%I:%M %p')
                                    string_response = "No classes for the rest of today, your next"
                                    string_response += " class is on {} in the ".format(day)
                                else:
                                    break

                        else:
                            response = "I do not have any classes for you"

                    if response != "":
                        myobj = gTTS(text=response, lang=language, slow=False)
                        myobj.save("Voice.mp3")
                        playsound("Voice.mp3")
                    else:
                        print("error")

        elif tag == "name_next_class":
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    for name in LIST_OF_STUDENTS:
                        if name in people:

                            try:
                                list_classes = json.load(open(PATH + "/" + name + "/Current.json"))
                            except:
                                string_response = "I do not have any classes for you"
                                continue

                            response = ""
                            string_response = "Your next class is {}"

                            curr_date = date.today()
                            day = calendar.day_name[curr_date.weekday()]

                            time_checked = datetime.today().strftime("%I:%M %p")
                            time_checked = datetime.strptime(time_checked,'%I:%M %p')

                            list_of_classes = list_classes['Current Classes']

                            day_count = curr_date.weekday()
                            while day_count <= 7:
                                for j in list_of_classes:
                                    if j['Time'] != 'Online':
                                        class_time = datetime.strptime(j['Time'],'%I:%M %p')
                                        if (day in j['Days']) and (time_checked <= class_time):
                                            response = string_response.format(j['Class'])
                                            break
                                if response == "":
                                    day_count += 1
                                    if day_count == 7:
                                        day_count = 0
                                    else:
                                        pass
                                    day = calendar.day_name[day_count]

                                    time_checked = datetime.strptime('1:00 AM','%I:%M %p')
                                    string_response = "No classes for the rest of today"
                                    string_response += ", your next class is {}"

                                else:
                                    break

                        else:
                            response = "I do not have any classes for you"

                    if response != "":
                        myobj = gTTS(text=response, lang=language, slow=False)
                        myobj.save("Voice.mp3")
                        playsound("Voice.mp3")
                    else:
                        print("error")

        elif tag == "next_event":
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:

                    # run the checker to get all events
                    events.create_events()

                    string_response = "West Georgia's next event is " # random.choice(i['responses']

                    cur_date = calendar.month_name[date.today().month] + " " + str(date.today().day)
                    time_now = datetime.strptime(datetime.now().strftime("%H:%M %p"), "%H:%M %p")

                    list_of_events = LIST_EVENTS['Events']

                    for j in list_of_events:
                        if cur_date == j['Date']:
                            if time_now <= datetime.strptime(j['Start Time'],'%I:%M %p'):
                                response = string_response + j['Name'] + " at " + j['Start Time']
                                break
                        else:
                            response = "There are no more events today"

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")

        else:
            list_of_intents = INTENTS['intents']

            for i in list_of_intents:
                if i['tag'] == tag:
                    response = random.choice(i['responses'])

                    myobj = gTTS(text=response, lang=language, slow=False)
                    myobj.save("Voice.mp3")
                    playsound("Voice.mp3")


def running(people):
    """
    Running loop for keyword detection to then begin the main Task loop.

    :param people: List of individuals in frame for use in personalized questions.
    """
    while True:

        print("standby")

        recorder = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recorder.listen(source)

        try:
            inpmain = recorder.recognize_google(audio)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            inpmain = " "
        except sr.RequestError as exception:
            print(f"Could get results from Google Speech Recognition; {0}".format(exception))
            inpmain = " "

        inpmain = inpmain.lower()

        if "quit" in inpmain:
            os.remove("Voice.mp3")
            break

        if "wolfie" in inpmain:
            chat(inpmain, people)
        else:
            pass


if __name__ == "__main__":
    running(['Steven Kight'])
