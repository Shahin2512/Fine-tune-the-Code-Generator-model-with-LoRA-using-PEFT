from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load dataset
dataset = load_dataset("json", data_files="data/code_dataset.json", split="train")

# Load tokenizer and model
model_id = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Apply LoRA
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

# Preprocess
def preprocess(examples):
    model_inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=1,
    fp16=True,
    logging_steps=10,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
model.save_pretrained("model/final_model")
tokenizer.save_pretrained("model/final_model")
