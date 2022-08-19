import sys
import time
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

import Tasks

LISTEN = True
PEOPLE = []
BACKGROUND = '#373737'

def main(people, inpmain):
   global PEOPLE
   PEOPLE = people

   inputed_text = inpmain

   root = Tk()
   root.title("Wolfie")

   captions = Label(text="", background=BACKGROUND)
   captions.place(relx=0.5, rely=0.75, anchor=CENTER)
   captions.config(font=('Helvetica bold',40), foreground='white')

   # Define Window Geometry
   root.lift()
   root.overrideredirect(1)
   root.overrideredirect(0)
   root.attributes('-alpha', 0.5)
   root.wm_attributes("-transparent", True)
   root.config(bg='systemTransparent')
   root.wm_attributes("-topmost", True)
   root.attributes('-topmost', True)
   root.configure(background=BACKGROUND)

   width, height = root.winfo_screenwidth(), root.winfo_screenheight()
   root.geometry('%dx%d+0+0' % (width,height))

   def task(inp, caption):
      response, bye = Tasks.chat(PEOPLE, inp)

      caption.config(text = response)
      root.update()

      global LISTEN
      if bye:
         time.sleep(3)
         root.withdraw()
         LISTEN = False
         return

   root.update()
   root.deiconify()
   while LISTEN:
      if inputed_text != "":
         task(inputed_text, captions)
         inputed_text = ""
      else:
         task(inputed_text, captions)
   
   root.update()
