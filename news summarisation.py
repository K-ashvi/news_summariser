from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

# Step 1: Load the Dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset['train'].select(range(2000))  # Reduced training data
val_data = dataset['validation'].select(range(500))  # Reduced validation data

# Step 2: Load Pre-Trained Tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Step 3: Preprocess Data
def preprocess_data(batch):
    inputs = tokenizer(batch["article"], max_length=256, truncation=True, padding="max_length")
    outputs = tokenizer(batch["highlights"], max_length=64, truncation=True, padding="max_length")
    inputs["labels"] = outputs["input_ids"]
    return inputs

train_data = train_data.map(preprocess_data, batched=True, remove_columns=["article", "highlights", "id"])
val_data = val_data.map(preprocess_data, batched=True, remove_columns=["article", "highlights", "id"])

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 4: Load Pre-Trained Model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bart-summarization",  
    evaluation_strategy="steps",        
    learning_rate=5e-5,                
    per_device_train_batch_size=4,     
    per_device_eval_batch_size=4,
    num_train_epochs=0.5,               # Shorter training duration
    gradient_accumulation_steps=4,      # Simulate larger batch size
    weight_decay=0.01,                 
    logging_dir="./logs",              
    save_total_limit=1,                 
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Model
model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")

print("Model training complete and saved to './fine_tuned_bart'.")
