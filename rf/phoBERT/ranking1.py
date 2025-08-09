# for english only
from sentence_transformers import CrossEncoder

# Load mô hình Cross-Encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "best laptop for gaming"
documents = [
    "This laptop has an RTX 3060 and is great for gaming.",
    "A budget laptop for office work.",
    "Gaming laptops with high refresh rate displays."
]
query = "Thủ đô Việt Nam ở đâu?"
documents = [
    "Hà Nội là thành phố trực thuộc trung ương có diện tích lớn thứ năm tại Việt Nam",
    "Câu lạc bộ bóng đá Hà Nội TNT.",
    "Hà Nội là thủ đô Việt Nam.",
    "Paris là thủ đô Pháp.",
    "HCM là thành phố  lớn nhất miền nam Việt Nam."
]
# Tạo list (query, doc) pairs
pairs = [(query, doc) for doc in documents]

# Dự đoán độ phù hợp
scores = model.predict(pairs)

# Sắp xếp theo điểm
ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

for doc, score in ranked_results:
    print(f"{score:.4f} - {doc}")
