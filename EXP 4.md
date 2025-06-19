<H3>ENTER YOUR NAME : SHARMITHA V</H3>
<H3>ENTER YOUR REGISTER NO : 212223110048</H3>
<H3>EX. NO.4</H3>
<H1 ALIGN =CENTER> Neural Machine Translation: English to French using Pre-trained Transformer</H1>

## AIM:
To implement neural machine translation from English to French using a pre-trained transformer model

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook


## ALGORITHM:

1.Install & import transformers and torch libraries.
2.Load pre-trained tokenizer and translation model (opus-mt-en-fr).
3.Tokenize the English input sentence.
4.Generate French translation using the model.
5.Decode and display the translated output.

##  PROGRAM:
~~~
!pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model and tokenizer for English-to-French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
import torch

def translate_text(text: str, max_length: int = 40) -> str:
    # Tokenize the input text and convert to input IDs
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate translation using the model
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated IDs back to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text
~~~
## OUTPUT:

![image](https://github.com/user-attachments/assets/4eab52b6-b46c-4f4c-ba9d-0dd3685107ec)


## RESULT:
The implementation of neural machine translation using a pre-trained transformer model was successfully executed, accurately translating English sentences into French.
