import pandas as pd
import numpy as np
import faiss
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os
import re
from collections import defaultdict
from tqdm import tqdm
warnings.filterwarnings('ignore')

class TicketClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(TicketClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class TicketDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class ComprehensiveRAGSystem:
    def __init__(self, csv_path=None):
        """Complete RAG system with training and evaluation"""
        self.csv_path = self._find_csv_path(csv_path)
        self.embedding_model = None
        self.llm_pipeline = None
        self.faiss_index = None
        self.ticket_data = None
        self.embeddings = None
        self.train_data = None
        self.test_data = None
        self.classifier = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up model directories
        self.model_dir = "C:/Users/2128124/OneDrive - Cognizant/Desktop/Vibe_Coding/Confidential-RAG/rag-model"
        self.index_dir = os.path.join(self.model_dir, "faiss_index")
        self.data_dir = os.path.join(self.model_dir, "data")
        self.classifier_dir = os.path.join(self.model_dir, "classifier")
        
        print("🚀 Initializing Complete RAG System with Evaluation...")
        print(f"Using device: {self.device}")
        
        # Create model directories if they don't exist
        for directory in [self.model_dir, self.index_dir, self.data_dir, self.classifier_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

    def _find_csv_path(self, csv_path):
        """Find CSV file automatically"""
        if csv_path and os.path.exists(csv_path):
            return csv_path
            
        possible_paths = [
            "C:/Users/2128124/OneDrive - Cognizant/Desktop/Vibe_Coding/data/solutions.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError("❌ Could not find solutions.csv file")
    
    def load_and_prepare_data(self):
        """Load and prepare data with train/test split"""
        print("📊 Loading and preparing data...")
        
        # Load CSV with error handling
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.ParserError:
            df = pd.read_csv(self.csv_path, on_bad_lines='skip', engine='python')
        
        # Clean data
        df = df[df['SNO'] != 'SNO'] if 'SNO' in df.columns else df
        df = df.dropna(subset=['Ticket Description', 'Ticket Solution'])
        
        # Ensure Domain column exists and consolidate domains
        if 'Domain' not in df.columns:
            df['Domain'] = 'General'
        df['Domain'] = df['Domain'].fillna('General')
        
        # Consolidate domains
        def consolidate_domain(domain):
            domain = str(domain).strip()
            if domain.startswith('HR'):
                return 'HR'
            elif domain.startswith('IT'):
                return 'IT'
            elif domain.startswith('Finance'):
                return 'Finance'
            elif domain.startswith('Learning'):
                return 'Learning'
            elif domain.startswith('Project'):
                return 'Project Management'
            else:
                return 'Other'
                
        df['Domain'] = df['Domain'].apply(consolidate_domain)
        
        # Enhanced text preprocessing
        def preprocess_text(text):
            text = str(text).strip()
            # Remove multiple spaces
            text = ' '.join(text.split())
            # Ensure minimum length
            if len(text) < 10:
                return None
            return text
            
        df['Ticket Description'] = df['Ticket Description'].apply(preprocess_text)
        df['Ticket Solution'] = df['Ticket Solution'].apply(preprocess_text)
        
        # Remove rows with invalid text
        df = df.dropna(subset=['Ticket Description', 'Ticket Solution'])
        
        # Create enhanced combined text for retrieval
        df['combined_text'] = (
            "Category: " + df['Domain'].astype(str) + 
            " | Issue: " + df['Ticket Description'].astype(str) + 
            " | Resolution: " + df['Ticket Solution'].astype(str)
        )
        
        # Print domain distribution
        domain_counts = df['Domain'].value_counts()
        print("\nDomain distribution:")
        print(domain_counts)
        
        # Train/test split for evaluation
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Domain'])
        except ValueError as e:
            print("\nWarning: Could not perform stratified split. Falling back to random split.")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        self.ticket_data = df.reset_index(drop=True)
        self.train_data = train_df.reset_index(drop=True)
        self.test_data = test_df.reset_index(drop=True)
        
        print(f"\n✅ Total tickets: {len(df)}")
        print(f"📚 Training set: {len(train_df)}")
        print(f"🧪 Test set: {len(test_df)}")
        
        # Print final domain distribution
        print("\nFinal domain distribution after preprocessing:")
        print(df['Domain'].value_counts())
    
    def setup_models(self):
        """Setup embedding and generation models"""
        print("🔤 Loading embedding model...")
        # Use a more powerful model for better semantic understanding
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        
        print("🤖 Loading generation model...")
        model_name = "microsoft/DialoGPT-medium"  # Use medium model for better responses
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,  # Increased for more detailed responses
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print("✅ Models loaded successfully!")
    
    def train_classifier(self, train_embeddings, train_labels, val_embeddings=None, val_labels=None):
        """Train neural classifier with dropout"""
        print("\n🧠 Training neural classifier...")
        
        # Prepare data
        num_classes = len(set(train_labels))
        input_dim = train_embeddings.shape[1]
        hidden_dim = 256
        
        # Initialize model
        self.classifier = TicketClassifier(input_dim, hidden_dim, num_classes).to(self.device)
        
        # Create data loaders
        train_dataset = TicketDataset(train_embeddings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        if val_embeddings is not None and val_labels is not None:
            val_dataset = TicketDataset(val_embeddings, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        num_epochs = 50
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.classifier.train()
            total_loss = 0
            
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.classifier(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            if val_embeddings is not None and val_labels is not None:
                self.classifier.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_embeddings, batch_labels in val_loader:
                        batch_embeddings = batch_embeddings.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = self.classifier(batch_embeddings)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.classifier.state_dict(), os.path.join(self.classifier_dir, "classifier.pt"))
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        
        # Load best model if validation was used
        if val_embeddings is not None and val_labels is not None:
            self.classifier.load_state_dict(torch.load(os.path.join(self.classifier_dir, "classifier.pt")))
        
        print("✅ Neural classifier training complete!")

    def build_vector_database(self):
        """Build FAISS vector database from training data"""
        print("🗄️ Building FAISS vector database...")
        
        # Generate embeddings for training data
        train_texts = self.train_data['combined_text'].tolist()
        self.embeddings = self.embedding_model.encode(train_texts, show_progress_bar=True)
        
        # Create FAISS index with cosine similarity
        dimension = self.embeddings.shape[1]
        faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
        
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine for normalized
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        # Train neural classifier
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        train_labels = self.label_encoder.fit_transform(self.train_data['Domain'])
        
        # Split training data for validation
        train_emb, val_emb, train_lab, val_lab = train_test_split(
            self.embeddings, train_labels, test_size=0.1, random_state=42, stratify=train_labels
        )
        
        # Train classifier
        self.train_classifier(train_emb, train_lab, val_emb, val_lab)
        
        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def retrieve_similar_tickets(self, query, top_k=5):
        """Retrieve similar tickets using FAISS and neural classifier"""
        # Preprocess query
        query = " ".join(query.split())  # Normalize spaces
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Get neural classifier prediction
        with torch.no_grad():
            query_tensor = torch.FloatTensor(query_embedding).to(self.device)
            pred_logits = self.classifier(query_tensor)
            pred_probs = torch.softmax(pred_logits, dim=1)
            predicted_domain = self.label_encoder.inverse_transform([pred_logits.argmax().item()])[0]
        
        # Search with higher k to filter results
        k_search = min(top_k * 2, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), k_search)
        
        # Get results and filter by similarity threshold and predicted domain
        results = []
        similarity_threshold = 0.5
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity < similarity_threshold:
                continue
                
            if idx < len(self.train_data):
                ticket = self.train_data.iloc[idx]
                # Boost similarity score if domain matches prediction
                domain_boost = 0.1 if ticket['Domain'] == predicted_domain else 0
                results.append({
                    'similarity': float(similarity + domain_boost),
                    'domain': ticket['Domain'],
                    'description': ticket['Ticket Description'],
                    'solution': ticket['Ticket Solution']
                })
                
            if len(results) >= top_k:
                break
        
        # Sort results by boosted similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def generate_response(self, query, retrieved_tickets):
        """Generate response using retrieved context"""
        if not retrieved_tickets:
            return "No similar tickets found. Please contact support."
        
        # Create concise context
        context = f"Query: {query}\n\nSimilar issue: {retrieved_tickets[0]['description']}\nSolution: {retrieved_tickets[0]['solution']}\n\nRecommendation:"
        
        try:
            # Generate response
            response = self.llm_pipeline(
                context, 
                max_new_tokens=30,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            answer = generated_text[len(context):].strip()
            
            # Clean up the response
            answer = re.sub(r'\n+', ' ', answer)
            answer = re.sub(r'\s+', ' ', answer)
            
            # If generation failed, return the direct solution
            if not answer or len(answer) < 10:
                return retrieved_tickets[0]['solution']
            
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return retrieved_tickets[0]['solution']
    
    def evaluate_retrieval_metrics(self, top_k_values=[1, 3, 5]):
        """Evaluate Precision@K and Recall@K"""
        print("📊 Evaluating Retrieval Metrics...")
        
        results = {}
        
        for k in top_k_values:
            precision_scores = []
            recall_scores = []
            
            for _, test_ticket in self.test_data.iterrows():
                query = test_ticket['Ticket Description']
                true_domain = test_ticket['Domain']
                
                # Retrieve similar tickets
                retrieved = self.retrieve_similar_tickets(query, top_k=k)
                
                if not retrieved:
                    precision_scores.append(0)
                    recall_scores.append(0)
                    continue
                
                # Get all tickets from the same domain in training set (relevant items)
                relevant_tickets = self.train_data[self.train_data['Domain'] == true_domain]
                total_relevant = len(relevant_tickets)
                
                # Count how many retrieved tickets are from the same domain
                retrieved_domains = [ticket['domain'] for ticket in retrieved]
                relevant_retrieved = sum(1 for domain in retrieved_domains if domain == true_domain)
                
                # Precision@K: What fraction of retrieved items are relevant?
                precision_at_k = relevant_retrieved / k if k > 0 else 0
                precision_scores.append(precision_at_k)
                
                # Recall@K: What fraction of relevant items are retrieved?
                recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0
                recall_scores.append(recall_at_k)
            
            results[f'Precision@{k}'] = np.mean(precision_scores)
            results[f'Recall@{k}'] = np.mean(recall_scores)
        
        return results
    
    """def evaluate_generation_metrics(self, sample_size=50):
        print("📝 Evaluating Generation Metrics...")
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        # Sample test data for generation evaluation
        test_sample = self.test_data.sample(min(sample_size, len(self.test_data)), random_state=42)
        
        for _, test_ticket in test_sample.iterrows():
            query = test_ticket['Ticket Description']
            reference_solution = test_ticket['Ticket Solution']
            
            # Generate response
            retrieved = self.retrieve_similar_tickets(query, top_k=3)
            generated_response = self.generate_response(query, retrieved)
            
            # Calculate ROUGE scores
            scores = scorer.score(reference_solution, generated_response)
            
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Average scores
        avg_rouge_scores = {
            metric: np.mean(scores) for metric, scores in rouge_scores.items()
        }
        
        return avg_rouge_scores"""
    
    def comprehensive_evaluation(self):
        """Run complete evaluation with all metrics"""
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # Retrieval metrics
        retrieval_results = self.evaluate_retrieval_metrics()
        
        print("\n📊 RETRIEVAL METRICS:")
        for metric, score in retrieval_results.items():
            print(f"  {metric}: {score:.3f}")
        
        # Generation metrics
        #generation_results = self.evaluate_generation_metrics()
        
        #print("\n📝 GENERATION METRICS (ROUGE):")
        #for metric, score in generation_results.items():
        #    print(f"  {metric.upper()}: {score:.3f}")
        
        # Domain-wise performance
        print("\n🏷️ DOMAIN-WISE PERFORMANCE:")
        domain_performance = self._evaluate_domain_performance()
        for domain, metrics in domain_performance.items():
            print(f"  {domain}: Accuracy={metrics['accuracy']:.3f}, Count={metrics['count']}")
        
        # Overall summary
        overall_precision = retrieval_results['Precision@3']
        overall_recall = retrieval_results['Recall@3']
        #overall_rouge = generation_results['rouge1']
        
        # Calculate F1@3 score
        f1_at_3 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"  Precision@3: {overall_precision:.3f}")
        print(f"  Recall@3: {overall_recall:.3f}")
        print(f"  F1@3: {f1_at_3:.3f}")
        #print(f"  Generation Quality (ROUGE-1): {overall_rouge:.3f}")
        
        return {
            'retrieval': retrieval_results,
            #'generation': generation_results,
            'domain_performance': domain_performance
        }
    
    def _evaluate_domain_performance(self):
        """Evaluate performance per domain"""
        domain_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for _, test_ticket in self.test_data.iterrows():
            query = test_ticket['Ticket Description']
            true_domain = test_ticket['Domain']
            
            retrieved = self.retrieve_similar_tickets(query, top_k=1)
            predicted_domain = retrieved[0]['domain'] if retrieved else 'Unknown'
            
            domain_stats[true_domain]['total'] += 1
            if predicted_domain == true_domain:
                domain_stats[true_domain]['correct'] += 1
        
        # Calculate accuracy per domain
        domain_performance = {}
        for domain, stats in domain_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            domain_performance[domain] = {
                'accuracy': accuracy,
                'count': stats['total']
            }
        
        return domain_performance
    
    def demo_queries(self):
        """Demonstrate system with sample queries"""
        print("\n" + "="*60)
        print("🎮 DEMO QUERIES")
        print("="*60)
        
        sample_queries = [
            "I can't login to OneCognizant portal",
            "My timesheet is not getting approved",
            "VPN connection issues from home",
            "Need to update bank details for salary",
            "Training certificate missing from LMS"
        ]
        
        for query in sample_queries:
            print(f"\n❓ Query: {query}")
            
            retrieved = self.retrieve_similar_tickets(query, top_k=2)
            response = self.generate_response(query, retrieved)
            
            print(f"🎯 Top Match: [{retrieved[0]['domain']}] {retrieved[0]['description'][:50]}...")
            print(f"📋 Solution: {retrieved[0]['solution'][:60]}...")
            print(f"🤖 Generated: {response}")
            print(f"📊 Similarity: {retrieved[0]['similarity']:.3f}")
    
    def interactive_mode(self):
        """Interactive mode for real-time queries"""
        print("\n" + "="*60)
        print("🤖 INTERACTIVE MODE")
        print("="*60)
        print("\nType your questions below. Type 'exit' or 'quit' to end the session.")
        
        while True:
            try:
                # Get user input
                print("\n❓ Your question: ", end='')
                query = input().strip()
                
                # Check for exit command
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thank you for using the system. Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Get response
                print("\n🔍 Searching for relevant tickets...")
                retrieved = self.retrieve_similar_tickets(query, top_k=3)
                
                if not retrieved:
                    print("❌ No relevant tickets found. Please try rephrasing your question.")
                    continue
                
                # Show results
                print("\n📋 Top matches:")
                for i, ticket in enumerate(retrieved, 1):
                    print(f"\n{i}. Category: {ticket['domain']}")
                    print(f"   Similarity: {ticket['similarity']:.3f}")
                    print(f"   Description: {ticket['description'][:100]}...")
                    print(f"   Solution: {ticket['solution'][:150]}...")
                
                # Generate response
                print("\n🤖 Generated response:")
                response = self.generate_response(query, retrieved)
                print(f"{response}\n")
                
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again with a different question.")

    def save_models(self):
        """Save all models and data"""
        print("\n💾 Saving models and data...")
        try:
            # Save FAISS index
            index_path = os.path.join(self.index_dir, "faiss_index.bin")
            faiss.write_index(self.faiss_index, index_path)
            print(f"✓ Saved FAISS index to {index_path}")
            
            # Save training data
            train_path = os.path.join(self.data_dir, "train_data.pkl")
            test_path = os.path.join(self.data_dir, "test_data.pkl")
            self.train_data.to_pickle(train_path)
            self.test_data.to_pickle(test_path)
            print(f"✓ Saved training data to {self.data_dir}")
            
            # Save embeddings
            embeddings_path = os.path.join(self.data_dir, "embeddings.npy")
            np.save(embeddings_path, self.embeddings)
            print(f"✓ Saved embeddings to {embeddings_path}")
            
            # Save classifier if exists
            if self.classifier is not None:
                classifier_path = os.path.join(self.classifier_dir, "classifier.pt")
                torch.save(self.classifier.state_dict(), classifier_path)
                print(f"✓ Saved classifier to {classifier_path}")
                
            # Save label encoder
            if self.label_encoder is not None:
                encoder_path = os.path.join(self.classifier_dir, "label_encoder.pkl")
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                print(f"✓ Saved label encoder to {encoder_path}")
            
            print("\n✅ All models and data saved successfully!")
            return True
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False

    def load_models(self):
        """Load all models and data"""
        print("\n📂 Loading saved models and data...")
        try:
            # Define file paths
            index_path = os.path.join(self.index_dir, "faiss_index.bin")
            train_path = os.path.join(self.data_dir, "train_data.pkl")
            test_path = os.path.join(self.data_dir, "test_data.pkl")
            embeddings_path = os.path.join(self.data_dir, "embeddings.npy")
            classifier_path = os.path.join(self.classifier_dir, "classifier.pt")
            encoder_path = os.path.join(self.classifier_dir, "label_encoder.pkl")
            
            # Check if all required files exist
            required_files = [
                index_path,
                train_path,
                test_path,
                embeddings_path,
                classifier_path,
                encoder_path
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    print(f"❌ Missing required file: {file_path}")
                    return False
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(index_path)
            print(f"✓ Loaded FAISS index from {index_path}")
            
            # Load training data
            self.train_data = pd.read_pickle(train_path)
            self.test_data = pd.read_pickle(test_path)
            print(f"✓ Loaded training data from {self.data_dir}")
            
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            print(f"✓ Loaded embeddings from {embeddings_path}")
            
            # Load label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✓ Loaded label encoder from {encoder_path}")
            
            # Initialize and load classifier
            num_classes = len(self.label_encoder.classes_)
            input_dim = self.embeddings.shape[1]
            hidden_dim = 256
            self.classifier = TicketClassifier(input_dim, hidden_dim, num_classes).to(self.device)
            self.classifier.load_state_dict(torch.load(classifier_path))
            self.classifier.eval()
            print(f"✓ Loaded classifier from {classifier_path}")
            
            print("\n✅ All models and data loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def train_and_evaluate(self):
        """Main training and evaluation pipeline"""
        try:
            # Try to load existing models first
            if self.load_models():
                print("✅ Using existing models...")
                # Setup models (still needed for embedding_model and llm_pipeline)
                self.setup_models()
            else:
                print("🔄 Training new models...")
                # Step 1: Load and prepare data
                self.load_and_prepare_data()
                
                # Step 2: Setup models
                self.setup_models()
                
                # Step 3: Build vector database
                self.build_vector_database()
                
                # Save the models
                self.save_models()
            
            # Step 4: Comprehensive evaluation
            results = self.comprehensive_evaluation()
            
            # Step 5: Demo queries
            self.demo_queries()
            
            print("\n🎉 Training and Evaluation Complete!")
            return results
            
        except Exception as e:
            print(f"❌ Error in training/evaluation: {e}")
            raise

def main():
    """Main function"""
    print("🚀 Starting Comprehensive RAG System Training & Evaluation")
    
    # Initialize and run system
    rag_system = ComprehensiveRAGSystem()
    results = rag_system.train_and_evaluate()
    
    print(f"\n📊 Final Results Summary:")
    print(f"  Precision@3: {results['retrieval']['Precision@3']:.3f}")
    print(f"  Recall@3: {results['retrieval']['Recall@3']:.3f}")
    
    # Calculate and show F1 score
    p3 = results['retrieval']['Precision@3']
    r3 = results['retrieval']['Recall@3']
    f1_3 = 2 * (p3 * r3) / (p3 + r3) if (p3 + r3) > 0 else 0
    print(f"  F1@3: {f1_3:.3f}")
    
    # Start interactive mode
    print("\nStarting interactive mode...")
    rag_system.interactive_mode()

if __name__ == "__main__":
    main() 