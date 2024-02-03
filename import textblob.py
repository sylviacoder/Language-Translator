from tkinter import * 
import googletrans
import textblob 
from tkinter import ttk, messagebox
import pyttsx3

 #gui outlook
source = Tk()
source.title('Universal - Translator')
source.geometry("880x300")

#defining the translation setup
def translate_it():
    #delete any previous translateions
    translated_text.delete(1.0, END)
    try:
       #get languages from dictionary keys and get the from key
       for key, value in languages.items():
           if (value == original_combo.get()):
               from_language_key = key
        #get the to language key
       for key, value in languages.items():
           if (value == translated_combo.get()):
               to_language_key = key
        #turn original into textblob
       words = textblob.TextBlob(original_text.get(1.0, END))

        #To translate text
       words = words.translate(from_lang=from_language_key, to=to_language_key)
        #output translated text to screen
       translated_text.insert(1.0, words) 

       #to activate speech. ENGINE(initialize, pass and run)
       speech = pyttsx3.init()
       speech.say(words)
       speech.runAndWait()
       


    except Exception as e:
        messagebox.showerror("Translator", e)

#defining the clear function
def clear():
    original_text.delete(1.0, END)
    translated_text.delete(1.0, END)

 
#get language list form translator
languages = googletrans.LANGUAGES

#converting to list
language_list = list(languages.values())



#Text boxes
original_text = Text(source, height=10, width=40)
original_text.grid(row=0, column=0, pady=20, padx=10)

universaltranslate_button = Button(source, text="Translate", font=("New Times Roman", 25), command=translate_it)
universaltranslate_button.grid(row=0, column=1, padx=10)

translated_text = Text(source, height=10, width=40)
translated_text.grid(row=0, column=2, pady=20, padx=10) 

#content boxex(combo)
original_combo = ttk.Combobox(source, height=50, value=language_list)
original_combo.current(21)
original_combo.grid(row=1, column=0)

translated_combo = ttk.Combobox(source, width=50, value=language_list)
translated_combo.current(26)
translated_combo.grid(row=1, column=2)

#To clear button
clear_button = Button(source, text="Restart", command=clear)
clear_button.grid(row=2, column=1)


source.mainloop()





