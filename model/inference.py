from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model/final_model")
    model = AutoModelForSeq2SeqLM.from_pretrained("model/final_model")
    return tokenizer, model

def generate_output(prompt):
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        #outputs = model.generate(**inputs, max_new_tokens=100)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
