import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Make sure to download nltk resources if not already present
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# ==========================================
# 1. Text Preprocessing
# ==========================================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        # Remove special characters, punctuation, and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenization & Stop word removal & Lemmatization
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words
        ]
        return " ".join(cleaned_words)

# ==========================================
# 2. Dataset Definition
# ==========================================
class HealthcareDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize and encode for GPT-2
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 3. Model Architecture (MA-CBiLSTM)
# ==========================================
class CMSGPTNLP_MAH(nn.Module):
    def __init__(self, gpt2_model_name='gpt2', num_classes=9, cnn_filters=128, lstm_hidden=256, num_heads=8):
        super(CMSGPTNLP_MAH, self).__init__()
        
        # Pre-trained GPT-2 for Word Embedding
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        
        gpt2_hidden_size = self.gpt2.config.n_embd # usually 768 for gpt2 base
        
        # 1D Convolutional Layer for local feature extraction
        self.conv1d = nn.Conv1d(in_channels=gpt2_hidden_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Bidirectional LSTM for temporal/sequential dependencies
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                              num_layers=1, batch_first=True, bidirectional=True)
        
        bilstm_output_size = lstm_hidden * 2
        
        # Multi-Head Attention Mechanism
        self.multihead_attn = nn.MultiheadAttention(embed_dim=bilstm_output_size, num_heads=num_heads, batch_first=True)
        
        # Fully Connected Classifier
        self.fc = nn.Linear(bilstm_output_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        # 1. GPT-2 Embeddings
        gpt_outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = gpt_outputs.last_hidden_state
        
        # 2. Convolutional Layer
        embeddings = embeddings.transpose(1, 2)
        cnn_out = self.relu(self.conv1d(embeddings))
        
        # 3. Bi-LSTM
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.bilstm(cnn_out)
        
        # 4. Multi-Head Attention
        attn_out, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        
        # Global Average Pooling over the sequence length
        pooled_out = torch.mean(attn_out, dim=1)
        
        # 5. Classification
        out = self.dropout(pooled_out)
        logits = self.fc(out)
        
        return logits

# ==========================================
# 4. Training and Evaluation Pipeline
# ==========================================
def train_model(dataset_path='data/healthcare_documentation.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        # Map the 9 specialties from the Kaggle dataset to class indices (0-8)
        specialty_map = {
            'Cardiovascular / Pulmonary': 0, 'Cardiovascular /Pulmonary': 0,
            'Urology': 1, 
            'General Medicine': 2,
            'Surgery': 3, 
            'Radiology': 4, 
            'Orthopedic': 5,
            'Obstetrics / Gynecology': 6, 'Obstetrics /Gynecology': 6,
            'Neurology': 7, 
            'Gastroenterology': 8
        }
        
        # Assuming typical column names 'transcription' and 'medical_specialty'
        text_col = 'transcription' if 'transcription' in df.columns else df.columns[1]
        label_col = 'medical_specialty' if 'medical_specialty' in df.columns else df.columns[0]
        
        df = df.dropna(subset=[text_col, label_col])
        df['Class'] = df[label_col].str.strip().map(specialty_map)
        df = df.dropna(subset=['Class']) # Drop specialties outside the 9 classes
        df['Class'] = df['Class'].astype(int)
        
    else:
        print(f"WARNING: Dataset not found at '{dataset_path}'.")
        print("Using DUMMY DATA for demonstration. Please download the dataset and provide the correct path.")
        df = pd.DataFrame({
            'transcription': ["Patient has severe chest pain", "Brain scan shows no abnormalities", "Surgery was successful"],
            'Class': [0, 7, 3] 
        })
        text_col = 'transcription'
    
    # Preprocessing
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor()
    df['Cleaned_Transcription'] = df[text_col].apply(preprocessor.clean_text)
    
    texts = df['Cleaned_Transcription'].values
    labels = df['Class'].values
    
    # Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token 
    
    # DataLoaders
    train_dataset = HealthcareDataset(X_train, y_train, tokenizer)
    test_dataset = HealthcareDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize Model
    model = CMSGPTNLP_MAH(num_classes=9).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    epochs = 50 
    
    # Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        
    print("Training Complete!")
    return model

if __name__ == "__main__":
    train_model()
