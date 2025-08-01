import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BertConfig
import logging
from complete_rag_system import ComprehensiveRAGSystem
from transformers import pipeline
import os
import pickle
from peft import PeftModel, PeftConfig
import warnings
import re
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variables for classifier
CLASSIFIER_PATH = "qlora-bert-early-stop"
BASE_MODEL_NAME = "prajjwal1/bert-tiny"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_classifier():
    """Load the classifier model with caching"""
    try:
        # Load label encoder
        encoder_path = os.path.join(CLASSIFIER_PATH, "label_encoder.pkl")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
            
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        num_labels = len(label_encoder.classes_)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
        
        # Load base model with correct config
        config = BertConfig.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            config=config
        ).to(device)
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, CLASSIFIER_PATH)
        model.eval()
        
        logging.info(f"Classifier loaded successfully with {num_labels} classes: {label_encoder.classes_}")
        return model, tokenizer, label_encoder
        
    except Exception as e:
        logging.error(f"Error loading classifier: {str(e)}")
        st.error(f"Failed to load classifier: {str(e)}")
        return None, None, None

@st.cache_resource
def load_gpt2_model():
    """Load the GPT-2 model with caching"""
    try:
        model_path = "gpt2-saved"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Ensure the tokenizer has the necessary special tokens
        special_tokens = {
            'pad_token': '[PAD]',
            'eos_token': '</s>',
            'bos_token': '<s>',
            'unk_token': '[UNK]'
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading GPT-2 model: {str(e)}")
        return None, None

@st.cache_resource
def load_rag_system():
    """Load the RAG system with caching"""
    try:
        rag_system = ComprehensiveRAGSystem()
        if rag_system.load_models():
            rag_system.setup_models()
            return rag_system
        return None
    except Exception as e:
        logging.error(f"Error initializing RAG system: {str(e)}")
        return None

def process_gpt2_response(model, tokenizer, question, max_length=200):
    """Process GPT-2 response with proper formatting and parameters"""
    try:
        # Prepare input with proper formatting
        input_text = f"Question: {question}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response with carefully tuned parameters
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        
        # Decode and clean up the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        answer = response.split("Answer:", 1)[-1].strip()
        
        # Clean up the response
        answer = re.sub(r'\s+', ' ', answer)  # Remove extra whitespace
        answer = re.sub(r'\d{10,}', '', answer)  # Remove long number sequences
        answer = re.sub(r'[^\w\s\.,!?-]', '', answer)  # Remove unusual characters
        answer = answer.strip()
        
        if not answer or len(answer.split()) < 5:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        return answer
        
    except Exception as e:
        logging.error(f"Error in GPT-2 response processing: {str(e)}")
        return "I apologize, but there was an error processing your question. Please try again."

def process_rag_response(rag_system, user_question, retrieved_tickets, max_tokens=150):
    """Process RAG response with enhanced context and generation"""
    try:
        # Create enhanced context with more details
        context = (
            f"Query: {user_question}\n\n"
            f"Similar Issues Found:\n"
        )
        
        # Add context from multiple retrieved tickets
        for i, ticket in enumerate(retrieved_tickets[:2], 1):
            context += (
                f"\nIssue {i}:\n"
                f"Category: {ticket['domain']}\n"
                f"Problem: {ticket['description']}\n"
                f"Solution: {ticket['solution']}\n"
            )
        
        context += "\nRecommended Solution:"
        
        # Generate response with adjusted parameters
        response = rag_system.llm_pipeline(
            context,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=rag_system.llm_pipeline.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        
        # Extract the generated part after the context
        answer = generated_text[len(context):].strip()
        
        # If the answer is too short, use the solution from the best match
        if len(answer.split()) < 10:
            answer = (
                f"Based on similar cases, here's the recommended solution:\n\n"
                f"{retrieved_tickets[0]['solution']}"
            )
        
        return answer
        
    except Exception as e:
        logging.error(f"Error in RAG response processing: {str(e)}")
        return retrieved_tickets[0]['solution']

def classify_question(text, model, tokenizer, label_encoder):
    """Classify the input text using the loaded model"""
    try:
        if not all([model, tokenizer, label_encoder]):
            logging.warning("Classification models not properly loaded")
            return "general"
        
        # Prepare input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
        
        # Convert to label
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = predictions[0][predicted_class].item()
        
        # Log prediction details
        logging.info(f"Question: {text}")
        logging.info(f"Predicted class: {predicted_label} (confidence: {confidence:.2f})")
        
        # Map to response type
        if predicted_label.lower() == 'technical':
            return "technical"
        elif predicted_label.lower() == 'confidential':
            return "confidential"
        else:
            return "general"
            
    except Exception as e:
        logging.error(f"Error in classification: {str(e)}")
        return "general"

def main():
    st.title("Smart Ticket Assistant")
    st.write("Enter your question below:")
    
    # Load models
    model, tokenizer, label_encoder = load_classifier()
    
    # Text input
    user_question = st.text_input("Question:", key="question_input")
    
    if user_question:
        try:
            # Classify the question
            question_type = classify_question(user_question, model, tokenizer, label_encoder)
            
            # Display classification result
            st.info(f"Question classified as: {question_type}")
            
            if question_type == "general":
                st.warning("I apologize, but I am not able to assist with general queries. Please provide a technical or specific question.")
                
            elif question_type == "technical":
                st.info("Processing technical question...")
                gpt2_model, gpt2_tokenizer = load_gpt2_model()
                
                if gpt2_model and gpt2_tokenizer:
                    with st.spinner("Generating response..."):
                        response = process_gpt2_response(gpt2_model, gpt2_tokenizer, user_question)
                        st.success("🤖 Answer:")
                        st.write(response)
                else:
                    st.error("Error loading GPT-2 model. Please try again later.")
                    
            elif question_type == "confidential":
                st.info("Processing confidential question using RAG system...")
                rag_system = load_rag_system()
                
                if rag_system:
                    # Show processing status
                    with st.spinner("Searching for relevant information..."):
                        retrieved = rag_system.retrieve_similar_tickets(user_question, top_k=3)
                        
                    if retrieved:
                        # Show similar tickets found
                        st.write("📚 Found similar cases:")
                        for i, ticket in enumerate(retrieved[:2], 1):
                            with st.expander(f"Similar Case {i} (Similarity: {ticket['similarity']:.2f})"):
                                st.write(f"**Category:** {ticket['domain']}")
                                st.write(f"**Description:** {ticket['description']}")
                                st.write(f"**Solution:** {ticket['solution']}")
                        
                        # Generate and show response
                        with st.spinner("Generating response..."):
                            response = process_rag_response(rag_system, user_question, retrieved)
                            st.success("🤖 Recommended Solution:")
                            st.write(response)
                    else:
                        st.warning("No relevant information found for your query.")
                else:
                    st.error("Error loading RAG system. Please try again later.")
                    
        except Exception as e:
            error_message = f"Error processing question: {str(e)}"
            st.error(error_message)
            logging.error(error_message)

if __name__ == "__main__":
    main() 