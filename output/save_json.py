from idlelib.iomenu import encoding
from lib2to3.pgen2.grammar import opmap
import json
import ollama
prompt2 = "Given a list of action descriptions, extract structured lines in the format: Time | Person | Action. Use “—” if time is missing and “Person ?” if person is missing. Keep actions short (3-5 words). If multiple persons appear, output a separate line for each. Output only the structured list."
def intension_generator(model="llama3.2:3b", prompt="", data=""):
    to_model = prompt + "\n" + data
    resp = ollama.generate(model=model, prompt=to_model, options={"temperature":0})
    return resp['response']
with open("outputs/result.json", "r", encoding="utf-8") as f:
    a = json.load(f)
    sent = str(a["representatives"])
print(sent)
print(intension_generator(model="llama3.2:3b", prompt=prompt2, data=sent))