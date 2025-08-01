# Smart Ticket Assistant

A comprehensive ticket classification and response system that combines BERT-based classification, GPT-2 generation, and RAG (Retrieval-Augmented Generation) to provide intelligent responses to user queries.

## 🎯 Overview

This system is designed to handle different types of support tickets intelligently:

- **Technical Queries**: Handled by a fine-tuned GPT-2 model
- **Confidential Queries**: Processed using RAG system with secure data handling
- **General Queries**: Redirected with appropriate messaging

## 🏗️ Architecture

The system consists of three main components:

1. **Classification System**
   - Uses QLora-BERT model for efficient ticket classification
   - Trained to identify technical, confidential, and general queries
   - Optimized for low-latency inference

2. **Technical Query Handler (GPT-2)**
   - Fine-tuned GPT-2 model for technical responses
   - Configured for coherent and relevant answers
   - Includes response cleaning and formatting

3. **Confidential Query Handler (RAG)**
   - Retrieval-Augmented Generation system
   - FAISS vector database for efficient similarity search
   - Context-aware response generation

## 📁 Project Structure

```
Classification/
├── classifier.py             # Classification model code
Confidential-RAG/
├── complete_rag_system.py    # RAG system implementation
General-GPT    
├── gpt2-saved/              # Fine-tuned GPT-2 model
Front-End
├── streamlit_app.py         # Streamlit web interface
data/
├──solutions.csv        # Training data for solutions
├──tickets.csv          # Training data for tickets
requirements.txt         # Project dependencies
```

## 🚀 Setup and Installation

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Model Requirements**
   - Python version: 3.8-3.11 (NOT 3.12)
   - CUDA compatible GPU recommended but not required
   - Minimum 8GB RAM
   - ~2GB disk space for models

## 💻 Usage

1. **Start the Application**
   ```bash
   streamlit run Classification/streamlit_app.py
   ```

2. **Access the Interface**
   - Open your browser to `http://localhost:8501`
   - Enter your question in the text box
   - System will automatically:
     - Classify the question
     - Route to appropriate model
     - Generate relevant response

3. **Response Types**
   - Technical: Detailed technical solutions
   - Confidential: Secure, context-aware responses
   - General: Redirection message

## 🔧 Features

### Classification
- Real-time query classification
- Confidence score display
- Automatic routing to appropriate handler

### Technical Response Generation
- Coherent and relevant answers
- Response cleaning and formatting
- Error handling and fallbacks

### RAG System
- Similar case retrieval
- Context-aware response generation
- Expandable similar cases view
- Similarity score display

### UI Features
- Clean, intuitive interface
- Loading indicators
- Error messages
- Response formatting
- Expandable similar cases

## 📊 Performance

The system achieves:
- Classification accuracy: Based on trained model
- Response time: 2-5 seconds average
- RAG retrieval precision@3: Model dependent
- GPT-2 response quality: Optimized for technical accuracy

## 🔍 Logging

The system maintains comprehensive logs:
- Classification decisions
- Model loading status
- Error tracking
- Response generation details

Log file: `app.log` in the application directory

## ⚠️ Known Limitations

1. Python Version Compatibility
   - Requires Python 3.8-3.11
   - Not compatible with Python 3.12

2. Model Limitations
   - GPT-2: May occasionally generate generic responses
   - RAG: Requires relevant cases in database
   - Classification: Depends on training data quality

## 🛠️ Troubleshooting

1. **Model Loading Issues**
   - Check Python version compatibility
   - Verify model files presence
   - Check log file for specific errors

2. **Response Quality Issues**
   - GPT-2: Try rephrasing technical questions
   - RAG: Ensure similar cases exist in database
   - Classification: Check confidence scores

3. **Performance Issues**
   - Check available RAM
   - Verify GPU availability if expected
   - Monitor system resources

## 🔒 Security Considerations

1. Data Handling
   - Confidential data processed through RAG system
   - No external API calls for sensitive data
   - Local model inference only

2. Model Security
   - Models run locally
   - No cloud dependencies
   - Controlled data access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Streamlit for the web interface
- FAISS for vector similarity search 
