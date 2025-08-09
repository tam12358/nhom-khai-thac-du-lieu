from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ----------------------------
# 1. Sample Vietnamese corpus
# ----------------------------
documents = [
    "BERT là mô hình transformer cho xử lý ngôn ngữ tự nhiên.",
    "BM25 là một hàm xếp hạng dùng trong công cụ tìm kiếm để ước lượng độ liên quan.",
    "Các mô hình học sâu như BERT có thể cải thiện việc xếp hạng tìm kiếm.",
    "Python là ngôn ngữ lập trình tuyệt vời cho học máy.",
    "Elasticsearch sử dụng BM25 mặc định để xếp hạng văn bản."
]

query = "Cách dùng BERT để xếp hạng tìm kiếm?"

# Tokenize corpus for BM25 (simple whitespace split)
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------
# 2. First-stage retrieval with BM25
# ----------------------------
top_n = 3
top_docs = bm25.get_top_n(query.lower().split(), documents, n=top_n)

print("\nBM25 top results:")
for doc in top_docs:
    print("-", doc)

# ----------------------------
# 3. Load PhoBERT (fine-tuned model preferred)
# ----------------------------
# If you don't have a fine-tuned version, you must fine-tune on a relevance dataset first
model_name = "vinai/phobert-base"  # or "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# ----------------------------
# 4. PhoBERT scoring
# ----------------------------
def phobert_score(query, doc):
    inputs = tokenizer(query, doc, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.squeeze().item()

# ----------------------------
# 5. Re-rank results
# ----------------------------
scored_docs = [(doc, phobert_score(query, doc)) for doc in top_docs]
reranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

print("\nPhoBERT re-ranked results:")
for doc, score in reranked:
    print(f"{score:.4f} -> {doc}")
