import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

print("🤖 GPT-2 Text Generator")
print("📥 Downloading GPT-2 model (~500MB)...")

# ===============================
# 1. LOAD MODEL
# ===============================
model_name = "gpt2"  # You can use: gpt2, gpt2-medium, gpt2-large

print(f"Loading {model_name}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add proper padding token (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Resize model embeddings to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"✅ Model loaded! Using: {device}")
print(f"Parameters: {model.num_parameters():,}")

# ===============================
# 2. GENERATION FUNCTION
# ===============================
def generate_text(prompt, max_length=100, temperature=0.7, num_outputs=1):
    """
    Generate text from a prompt
    
    Args:
        prompt: Starting text
        max_length: How long the output should be
        temperature: Creativity (0.1=boring, 1.0=creative)
        num_outputs: How many versions to generate
    """
    
    # Tokenize input with proper attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=False,  # No padding for single input
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Proper attention mask
            max_length=len(inputs["input_ids"][0]) + max_length,
            temperature=temperature,
            num_return_sequences=num_outputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,  # Use proper pad token
            no_repeat_ngram_size=2,  # Avoid repetition
            repetition_penalty=1.1  # Reduce repetition
        )
    
    # Decode results
    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(text)
    
    return results


# ===============================
# 4. INTERACTIVE MODE
# ===============================
print("\n" + "=" * 50)
print("🎮 INTERACTIVE MODE")
print("=" * 50)
print("Type your prompt and press Enter!")
print("Commands: 'quit' to exit, 'help' for options")
print()

while True:
    try:
        user_input = input("Enter prompt: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'help':
            print("\n📋 Options:")
            print("- Just type text to continue it")
            print("- Try: 'The user reported that'")
            print("- Try: 'To fix this problem,'")
            print("- Try: 'Dear customer,'")
            continue
        
        if not user_input:
            continue
        
        print("\n🤖 Generating...")
        
        # Generate text
        results = generate_text(
            user_input, 
            max_length=80, 
            temperature=0.7,
            num_outputs=1
        )
        
        print(f"✨ Result:\n{results[0]}\n")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")

# ===============================
# 5. SAVE MODEL (OPTIONAL)
# ===============================
print("\n💾 Want to save model locally for faster loading?")
save = input("Save model? (y/n): ").lower()

if save == 'y':
    print("Saving model...")
    model.save_pretrained("./gpt2-saved")
    tokenizer.save_pretrained("./gpt2-saved")
    print("✅ Saved to ./gpt2-saved/")
    print("💡 Next time change model_name to './gpt2-saved'")

print("\n🎉 GPT-2 Text Generator Complete!") 