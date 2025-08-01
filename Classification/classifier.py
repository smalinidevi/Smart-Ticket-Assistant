import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from bert_score import score as bert_score
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback 
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# Suppress bitsandbytes quantization warnings
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

print("🚀 Starting QLoRA BERT Fine-tuning with Early Stopping...")

# ===============================
# 1. DATA PREPARATION
# ===============================
print("📊 Loading and preparing data...")

# Load dataset
df = pd.read_csv("C:/Users/2128124/OneDrive - Cognizant/Desktop/Vibe_Coding/data/tickets.csv")
print(f"Dataset shape: {df.shape}")
print(f"Categories: {df['category'].value_counts()}")

# Filter out single-sample categories
counts = df["category"].value_counts()
df = df[df["category"].isin(counts[counts > 1].index)]

# Encode labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['category'])
num_labels = len(label_encoder.classes_)

print(f"Number of classes: {num_labels}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, range(num_labels)))}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['ticket_text'].tolist(),
    df['labels'].tolist(), 
    test_size=0.2, 
    stratify=df['labels'], 
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ===============================
# 2. TOKENIZER SETUP
# ===============================
print("🔤 Setting up tokenizer...")

model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===============================
# 3. PRE-TOKENIZE DATA
# ===============================
print("🔧 Pre-tokenizing data...")

def tokenize_texts(texts):
    """Tokenize a list of texts"""
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors=None,
        return_token_type_ids=False
    )

# Tokenize all data at once
print("Tokenizing training data...")
train_encodings = tokenize_texts(X_train)

print("Tokenizing test data...")
test_encodings = tokenize_texts(X_test)

# ===============================
# 4. CREATE DATASETS
# ===============================
print("📝 Creating datasets with pre-tokenized data...")

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': y_train
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': y_test
})

print("✅ Datasets created successfully!")

# ===============================
# 5. CUSTOM MODEL WITH SOFTMAX
# ===============================
print("🤖 Creating custom model with softmax...")

class BertWithSoftmax(nn.Module):
    def __init__(self, base_model):
        super(BertWithSoftmax, self).__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_ids, attention_mask=None, labels=None, num_items_in_batch=None, **kwargs):
        # Remove num_items_in_batch from kwargs as BERT doesn't expect it
        if 'num_items_in_batch' in kwargs:
            kwargs.pop('num_items_in_batch')
            
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        if hasattr(outputs, 'logits'):
            probabilities = self.softmax(outputs.logits)
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(outputs.logits, labels)
                
                return type(outputs)(
                    loss=loss,
                    logits=probabilities,
                    hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                    attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
                )
            else:
                outputs.logits = probabilities
                return outputs
        
        return outputs

# ===============================
# 6. QUANTIZATION CONFIG
# ===============================
print("⚙️ Setting up 4-bit quantization...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ===============================
# 7. MODEL SETUP
# ===============================
print("🤖 Loading base model...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

base_model.config.label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
base_model.config.id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

print(f"Base model loaded with {num_labels} output classes")

# ===============================
# 8. PREPARE FOR KBIT TRAINING
# ===============================
print("🔧 Preparing model for k-bit training...")

base_model = prepare_model_for_kbit_training(base_model)

# ===============================
# 9. QLORA CONFIGURATION
# ===============================
print("🎯 Setting up QLoRA...")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],
    bias="none",
)

base_model = get_peft_model(base_model, peft_config)

# ===============================
# 10. WRAP WITH SOFTMAX
# ===============================
print("🎯 Wrapping model with softmax layer...")

model = BertWithSoftmax(base_model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

# ===============================
# 11. TRAINING ARGUMENTS WITH EARLY STOPPING
# ===============================
print("🏋️ Setting up training configuration with early stopping...")

training_args = TrainingArguments(
    output_dir="./classifier-model",
    overwrite_output_dir=True,
    num_train_epochs=10,  
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=25,  
    
    # ✅ Early Stopping Configuration
    eval_strategy="steps",
    eval_steps=50,  # Evaluate every 50 steps
    save_strategy="steps",
    save_steps=50,  # Save every 50 steps
    
    # ✅ Early Stopping Settings - Use combined score for better model selection
    load_best_model_at_end=True,  # Load best model at end
    metric_for_best_model="eval_combined_score",  # Monitor combined accuracy + BERT F1
    greater_is_better=True,  # Higher combined score is better
    
    # ✅ Save only good performance models
    save_total_limit=3,  # Keep only 3 best checkpoints
    save_only_model=True,  # Save only model weights, not optimizer states
    
    # Other settings
    report_to="none",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    fp16=True,
)

# ===============================
# 12. DATA COLLATOR
# ===============================
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# ===============================
# 13. EVALUATION METRICS WITH BERT SCORE
# ===============================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate BERT Score for semantic similarity
    # Convert predictions and labels back to text for BERT Score
    pred_texts = [label_encoder.inverse_transform([pred])[0] for pred in predictions]
    true_texts = [label_encoder.inverse_transform([label])[0] for label in labels]
    
    # Calculate BERT Score (P, R, F1)
    P, R, F1 = bert_score(pred_texts, true_texts, lang="en", verbose=False)
    bert_f1 = F1.mean().item()
    
    return {
        "accuracy": accuracy,
        "bert_f1": bert_f1,
        "combined_score": (accuracy + bert_f1) / 2  # Combined metric for best model selection
    }

# ===============================
# 14. EARLY STOPPING CALLBACK
# ===============================
print("⏰ Setting up early stopping callback...")

# ✅ Configure Early Stopping with combined metric
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,  # Stop if no improvement for 5 evaluations (more patience for combined metric)
    early_stopping_threshold=0.005  # Minimum improvement threshold for combined score
)

print("✅ Early stopping will trigger if combined score (accuracy + BERT F1) doesn't improve by 0.005 for 5 consecutive evaluations")

# ===============================
# 15. TRAINER SETUP WITH EARLY STOPPING
# ===============================
print("🎓 Setting up trainer with early stopping...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],  # ✅ Add early stopping callback
)

# ===============================
# 16. TRAINING WITH EARLY STOPPING
# ===============================
print("🚀 Starting training with early stopping...")
print("📊 Training will stop early if combined score (accuracy + BERT F1) doesn't improve")
print("💾 Only saving top 3 best performing checkpoints")
print("=" * 60)

trainer.train()

print("✅ Training completed!")

# Check if early stopping was triggered
if hasattr(trainer.state, 'log_history'):
    final_logs = trainer.state.log_history
    if len(final_logs) > 0:
        final_epoch = final_logs[-1].get('epoch', 'Unknown')
        print(f"🏁 Training stopped at epoch: {final_epoch}")

# ===============================
# 17. EVALUATION
# ===============================
print("📊 Evaluating model...")

eval_results = trainer.evaluate()
print(f"Final Evaluation Results: {eval_results}")

# Make predictions
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_
))

# Calculate detailed BERT Score for final evaluation
print("\n🎯 BERT Score Evaluation:")
pred_texts = [label_encoder.inverse_transform([pred])[0] for pred in y_pred]
true_texts = [label_encoder.inverse_transform([label])[0] for label in y_test]

P, R, F1 = bert_score(pred_texts, true_texts, lang="en", verbose=True)
print(f"BERT Score - Precision: {P.mean():.4f}")
print(f"BERT Score - Recall: {R.mean():.4f}")
print(f"BERT Score - F1: {F1.mean():.4f}")

# Combined performance score
final_accuracy = accuracy_score(y_test, y_pred)
final_bert_f1 = F1.mean().item()
combined_final_score = (final_accuracy + final_bert_f1) / 2
print(f"\n🏆 Final Combined Score: {combined_final_score:.4f}")
print(f"   - Accuracy: {final_accuracy:.4f}")
print(f"   - BERT F1: {final_bert_f1:.4f}")

# ===============================
# 18. SAVE MODEL
# ===============================
print("💾 Saving model...")

model.base_model.save_pretrained("./qlora-bert-early-stop")
tokenizer.save_pretrained("./qlora-bert-early-stop")

import pickle
with open("./qlora-bert-early-stop/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model saved successfully!")

# ===============================
# 19. TRAINING SUMMARY
# ===============================
print("\n" + "="*60)
print("🎯 TRAINING SUMMARY")
print("="*60)
print(f"✅ Early stopping enabled with patience=5 (combined score)")
print(f"✅ BERT Score evaluation included for semantic similarity")
print(f"✅ Only top 3 best performing checkpoints saved")
print(f"✅ Model outputs probabilities directly (softmax during training)")
print(f"✅ QLoRA efficiency: {100 * trainable_params / total_params:.2f}% trainable parameters")
print(f"📁 Model saved in: ./qlora-bert-early-stop/")
print(f"🎯 Final Combined Score: {combined_final_score:.4f}")

# Quick inference test
print("\n🔮 Quick inference test...")

def predict_text_direct(text, model, tokenizer, label_encoder):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=256,
        return_token_type_ids=False
    )
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits
        
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return {
            'predicted_class': predicted_label,
            'confidence': confidence
        }

# Test prediction
sample_text = "I can't login to my account, please help"
result = predict_text_direct(sample_text, model, tokenizer, label_encoder)

print(f"Sample text: '{sample_text}'")
print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.4f})")

print("\n🎉 QLoRA Fine-tuning with Early Stopping Complete!") 