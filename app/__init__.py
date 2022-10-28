from flask import Flask, render_template, request, session, redirect

#Import main library
import gdown
import torch
import transformers
import json
import random
import os
import math
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import pip
import sys
import subprocess
#from txtai.pipeline import Translation




print(transformers.__version__)
#initialise the app
app = Flask(__name__)

#import the paraphrasing models
#from transformers import MarianMTModel, MarianTokenizer
#en_model_name = 'Ghani-25/parapred_en_fr'
#en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
#en_model = MarianMTModel.from_pretrained(en_model_name)
#target_model_name = 'Ghani-25/parapred_fr_en'
#target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
#target_model = MarianMTModel.from_pretrained(target_model_name)

#url = "https://github.com/Ghani-25/waapred/raw/master/BESTmodel_weights.pt"
#output = "BESTmodel_weights.pt"
#gdown.download(url, output, quiet=False)


#create our "home" route using the "index.html" page

@app.route("/")
def index():
    return render_template("index.html")

#Set a post method to yield predictions on page

@app.route('/predict', methods = ['POST', 'GET'])

def predict():
    #if transformers.__version__=='4.1.1':
    #    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==4.21.3'])

    url = "https://drive.google.com/uc?export=download&id=1rBG3CI5b7uG90TOX7c4mJytdPF560M_F"
    output = "BESTmodel_weights.pt"
    gdown.download(url, output, quiet=False)
    urll = './BESTmodel_weights.pt'
    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")        
    modell = torch.load(urll, map_location=torch.device('cpu'))
    modell.to(device)
    modell.eval()  

    BASE_MODEL = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    y_preds = []
    contenu = request.form.get("comment")
    encoded = tokenizer(request.form.get("comment"), truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cpu")
    y_preds += modell(**encoded).logits.reshape(-1).tolist()

    indications = ["Rédiger un message compris entre 100 et 150 caractères", "Mettre la phrase d'accroche en avant", "S'adresser à la personne avec son prénom/nom"]
    if y_preds[0] <= 20.60 :
        realvalue = y_preds[0]
        realone = f'Le taux de prédiction est compris entre {realvalue-((realvalue*11)/100)} et {realvalue+((realvalue*11)/100)}, pensez à modifier votre message en considérant les indications suivantes {*indications,}'
    elif y_preds[0] > 20.60 and y_preds[0] < 22.99 :
        realvalue = y_preds[0] * 2.11
        realone = f'Le taux de prédiction est compris entre {realvalue-((realvalue*11)/100)} et {realvalue+((realvalue*11)/100)}, votre message peut etre amélioré en considérant les indications suivantes {*indications,}'
    else:
        realvalue = y_preds[0] * 3.11
        realone = f'Le taux de prédiction est compris entre {realvalue-((realvalue*11)/100)} et {realvalue+((realvalue*11)/100)}, votre message semblerait être bon :)'

    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame([request.form.get("comment"), realone], ["CONTENT", "Prediction"]).T
    return render_template('simple.html',  tables=[df.to_html(classes='data')], titles=request.form.get("comment"), header="true")
    #return df.to_json(orient = 'records')
    """
@app.route('/predict/paraphrase', methods = ['POST', 'GET'])
def paraphraser ():

    translate = Translation()
    languages = ["es"]
    en_texts = request.form.get("comment")

    translations = [translate(en_texts, language) for language in languages]

    english = translate(translations, "fr")

    for x, text in enumerate(translations):
        #print("Original Language: %s" % languages[x])
        #print("Translation: %s" % text)
        return("Paraphrases: %s" % english[0:5] )
        print()
    
    #if transformers.__version__=='4.21.3':
    #    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==4.1.1'])
    

    en_texts = [request.form.get("comment")]
    en_textss = [request.form.get("comment")]
    def translate(texts, model, tokenizer, language="fr"):
        # Prepare the text data into appropriate format for the model
        template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
        src_texts = [template(text) for text in texts]

        # Tokenize the texts
        encoded = tokenizer.prepare_seq2seq_batch([src_texts[0]], return_tensors='pt')
        
        # Generate translation using model
        translated = model.generate(**encoded)

        # Convert the generated tokens indices back into text
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return translated_texts
        
    def back_translate(texts, source_lang="en", target_lang="fr"):
        # Translate from source to target language
        fr_texts = translate(texts, target_model, target_tokenizer, 
                            language=target_lang)

        # Translate from target language back to source language
        back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                        language=source_lang)
        
        return back_translated_texts
    
    aug_texts = back_translate(en_texts, source_lang="fr", target_lang="en")    
    #return(aug_texts[0])
    
    #pd.set_option('display.max_rows', 500)
    #df = pd.DataFrame([en_textss, aug_texts[0]], ["CONTENT", "Paraphrased"]).T
    #print(df)
    return render_template('paraphrasedpage.html',  content=aug_texts[0])
    #return(df.to_html())
    """

#@app.route('/predictparaphrase', methods = ['POST', 'GET'])
#def predictpara():
    #url = "https://drive.google.com/uc?export=download&id=1rBG3CI5b7uG90TOX7c4mJytdPF560M_F"
    #output = "BESTmodel_weights.pt"
    #gdown.download(url, output, quiet=False)
    #urll = './BESTmodel_weights.pt'
    #device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")        
    #modell = torch.load(urll, map_location=torch.device('cpu'))
    #modell.to(device)
    #modell.eval() 
    #BASE_MODEL = "camembert-base"
    #tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    #y_preds = []
    #encoded = tokenizer(request.form.get("comment"), truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cpu")
    #y_preds += modell(**encoded).logits.reshape(-1).tolist()
    #y_predss = y_preds[0]
    #x = random.randint(25,50)

    #pd.set_option('display.max_rows', 500)
    #df = pd.DataFrame([request.form.get("comment"), y_predss+(y_predss*(x/100))], ["CONTENT", "Prediction"]).T
    #return render_template('simple.json',  tables=[df.to_json(orient = 'records')], titles=df.columns.values)
    ##return df.to_json(orient = 'records')

