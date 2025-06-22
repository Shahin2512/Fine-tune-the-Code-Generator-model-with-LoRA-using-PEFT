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

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=16,  # Increased rank
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q", "v"]  # Specify target modules for CodeT5
)
model = get_peft_model(model, peft_config)

# Print trainable parameters
model.print_trainable_parameters()

# Preprocess function
def preprocess(examples):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Split dataset for validation
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # Increased epochs
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    logging_steps=50,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=100,
    learning_rate=3e-4,  # Adjusted learning rate
    predict_with_generate=True,
    generation_max_length=128,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    padding=True
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("model/final_model")
tokenizer.save_pretrained("model/final_model")