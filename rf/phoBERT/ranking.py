from torch import softmax
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")


# Sample query and documents
query = "Thủ đô Việt Nam ở đâu?"
documents = [
    "Hà Nội là thành phố trực thuộc trung ương có diện tích lớn thứ năm tại Việt Nam",
    "Câu lạc bộ bóng đá Hà Nội TNT.",
    "Hà Nội là thủ đô Việt Nam.",
    "Paris là thủ đô Pháp.",
    "HCM là thành phố  lớn nhất miền nam Việt Nam."
]

# 1 label for ranking score

# Tokenize query and documents
inputs = tokenizer([query] * len(documents), documents, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs).logits

print(outputs)
# Convert logits to ranking scores using softmax
scores = softmax(outputs, dim=0).squeeze().tolist()

# Rank documents based on scores
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=False)

# Display ranked documents
for item in enumerate(ranked_docs):
    print(item)

