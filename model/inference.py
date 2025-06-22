from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model/final_model")
    model = AutoModelForSeq2SeqLM.from_pretrained("model/final_model")
    return tokenizer, model

def generate_output(prompt):
    tokenizer, model = load_model()
    
    # Ensure the input is properly formatted
    # The model was trained on direct prompts, so use the prompt as-is
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128  # Match training max_length
    )
    
    # Move inputs to the same device as model
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Increased from 100
            do_sample=False,
            num_beams=4,  # Add beam search for better quality
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated tokens (exclude input)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # For seq2seq models, the output already excludes the input
    return generated_text
