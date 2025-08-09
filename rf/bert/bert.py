# python 3.9
# pip install transformers
# pip install torch
#!pip3 install fairseq
#!pip3 install fastbpe
#!pip3 install vncorenlp
#!pip3 install transformers
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Sample query and documents
query = "What is the capital of France?"
documents = [
    "I love Paris.",
    "Berlin is a city in Germany.",
    "Paris is the capital of France.",
    "France is known for its Eiffel Tower located in Paris."
]

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)  
# 1 label for ranking score

# Tokenize query and documents
inputs = tokenizer([query] * len(documents), documents, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs).logits

# Convert logits to ranking scores using softmax
scores = softmax(outputs, dim=0).squeeze().tolist()

# Rank documents based on scores
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# Display ranked documents
for idx, (doc, score) in enumerate(ranked_docs):
    print(f"Rank {idx + 1}: {doc} (Score: {score:.4f})")
