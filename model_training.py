from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch

# huggingface turkish bert
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-cased")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-turkish-128k-cased")

#huggingface savasy model bilgi universitesi savas yıldırım


model_savasy = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
tokenizer_savasy = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")

#label_list = [
#    "O",       # Outside of a named entity
#    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
#    "I-MISC",  # Miscellaneous entity
#    "B-PER",   # Beginning of a person's name right after another person's name
#    "I-PER",   # Person's name
#    "B-ORG",   # Beginning of an organisation right after another organisation
#    "I-ORG",   # Organisation
#    "B-LOC",   # Beginning of a location right after another location
#    "I-LOC"    # Location
#]
#
#sequence = "Olası Irak operasyonunda Türkiye’den taleplere ilişkin son kararın alınacağı MGK öncesinde Ankara’ya gelen ABD’niniki bakan yardımcısı yirmi milyar dolarlık paket açacak."
#
#tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
#inputs = tokenizer.encode(sequence, return_tensors="pt")
#
#outputs = model(inputs)[0]
#predictions = torch.argmax(outputs, dim=2)
#
#print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])
