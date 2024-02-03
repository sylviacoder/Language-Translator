import tkinter as tk
from tkinter import messagebox
from textblob import TextBlob
import googletrans

def populate_combo_box(combo_box, languages):
    combo_box['values'] = list(languages.values())

def translate_text(original_text: str, from_language_key: str, to_language_key: str) -> str:
    try:
        words = TextBlob(original_text)
        words = words.translate(from_lang=from_language_key, to=to_language_key)
        return str(words)
    except Exception as e:
        messagebox.showerror("Translator", e)
        return ""

def handle_translation():
    original_text = original_text_box.get("1.0", tk.END)
    if not original_text.strip():
        messagebox.showerror("Translator", "Please enter text to translate.")
        return

    from_language_key = None
    to_language_key = None

    for key, value in languages.items():
        if value == from_language.get():
            from_language_key = key
        if value == to_language.get():
            to_language_key = key

    if not from_language_key or not to_language_key:
        messagebox.showerror("Translator", "Please select a valid language.")
        return

    translated_text = translate_text(original_text, from_language_key, to_language_key)
    translated_text_box.delete("1.0", tk.END)
    translated_text_box.insert("1.0", translated_text)

def clear():
    original_text_box.delete("1.0", tk.END)
    translated_text_box.delete("1.0", tk.END)

languages = googletrans.LANGUAGES
language_list = list(languages.values())

root = tk.Tk()

original_frame = tk.Frame(root)
original_frame.pack(pady=10)

original_label = tk.Label(original_frame, text="Original Text:")
original_label.grid(row=0, column=0, padx=(20, 0))
original_text_box = tk.Text(original_frame, height=10, width=50)
original_text_box.grid(row=0, column=1, padx=(10, 0))

from_language_frame = tk.Frame(root)
from_language_frame.pack(fill=tk.X, pady=10)

from_language_label = tk.Label(from_language_frame, text="From Language:")
from_language_label.grid(row=0, column=0, padx=(20, 0))
from_language = tk.StringVar(from_language_frame)
from_language.set(language_list[0])
from_language_combo = tk.OptionMenu(from_language_frame, from_language, *language_list)
from_language_combo.grid(row=0, column=1, padx=(10, 0))

to_language_frame = tk.Frame(root)
to_language_frame.pack(fill=tk.X, pady=10)

to_language_label = tk.Label(to_language_frame, text="To Language:")
to_language_label.grid(row=0, column=0, padx=(20, 0))
to_language = tk.StringVar(to_language_frame)
to_language.set(language_list[1])
to_language_combo = tk.OptionMenu(to_language_frame, to_language, *language_list)
to_language_combo.grid(row=0, column=1, padx=(10, 0))

translate_frame = tk.Frame(root)
translate_frame.pack(pady=10)

translate_button = tk.Button(translate_frame, text="Translate", command=handle_translation)
translate_button.grid(row=0, column=0, padx=(20, 0))

clear_button = tk.Button(translate_frame, text="Clear", command=clear)
clear_button.grid(row=0, column=1, padx=(10, 0))

translated_frame = tk.Frame(root)
translated_frame.pack(fill=tk.BOTH, expand=True, pady=10)

translated_label = tk.Label(translated_frame, text="Translated Text:")
translated_label.grid(row=0, column=0, padx=(20, 0))
translated_text_box = tk.Text(translated_frame, height=10, width=50)
translated_text_box.grid(row=0, column=1, padx=(10, 0))

populate_combo_box(from_language_combo, languages)
populate_combo_box(to_language_combo, languages)

root.mainloop()