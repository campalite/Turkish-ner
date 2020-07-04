from model_training import model, tokenizer, model_savasy, tokenizer_savasy
from transformers import pipeline
import torch
import time

label_list = [
        "O",       # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",   # Beginning of a person's name right after another person's name
        "I-PER",   # Person's name
        "B-ORG",   # Beginning of an organisation right after another organisation
        "I-ORG",   # Organisation
        "B-LOC",   # Beginning of a location right after another location
        "I-LOC"    # Location
    ]
def modelTrain(func):
    def wrapper():
        startTime = time.perf_counter()
        func()
        processTime = time.perf_counter() - startTime
        print("Process Time: " + str(processTime))
    return wrapper
    
    
def huggingFace():
     
    sequence = "Ayşe, pazartesi günü Fatma'yı sekreter olarak atadı. Onu seçti çünkü eski bir çalışan olarak dışişleri deneyimi vardı."
    #sequence = "Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı."
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    inputs = tokenizer.encode(sequence, return_tensors="pt")
    
    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, dim=2)
    
    print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])

def huggingFace_Savasy():
    #sequence = "Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı."
    sequence = "Ayşe, pazartesi günü Fatma'yı sekreter olarak atadı. Onu seçti çünkü eski bir çalışan olarak dışişleri deneyimi vardı."
    ner_savasy = pipeline('ner',model = model_savasy, tokenizer = tokenizer_savasy)
    ner_savasy(sequence)
    tokens = ner_savasy.tokenizer.tokenize(ner_savasy.tokenizer.decode(ner_savasy.tokenizer.encode(sequence)))
    inputs = ner_savasy.tokenizer.encode(sequence, return_tensors = "pt")
    outputs = model(inputs)[0]
    
    predictions = torch.argmax(outputs, dim=2)
    print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])
    
def savasy_exam():
    
    ner=pipeline('ner', model=model_savasy, tokenizer=tokenizer_savasy)
    sentence = "Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı."
    ner(sentence)
    #result = []
    tokens = ner.tokenizer.tokenize(ner.tokenizer.decode(ner.tokenizer.encode(sentence)))
    inputs = ner.tokenizer.encode(sentence, return_tensors = "pt")
    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, dim = 2)
    print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])
    #for x in label_list:
    #    result.append(ner(sentence)[0])
    #result.append(result)
    #result = pd.DataFrame(result)
    #result.head()
    #print(result)
    
#huggingFace = modelTrain(huggingFace)
#huggingFace()    
    
    
#huggingFace_Savasy = modelTrain(huggingFace_Savasy)
#huggingFace_Savasy()
    
    
#savasy_exam = modelTrain(savasy_exam)
#savasy_exam()