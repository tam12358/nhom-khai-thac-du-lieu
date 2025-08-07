import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from wordcloud import WordCloud
from underthesea import text_normalize
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance

# Chọn danh sách các chủ đề của tài liệu/văn bản cho thử nghiệm
topics = [
    'the-thao',
    'giao-duc',
    'khoa-hoc'
]

# Tạo một tập dữ liệu thử nghiệm gồm các tài liệu/văn bản thuộc về 2-3 chủ đề
# Cấu trúc dữ liệu dạng list - lưu thông tin danh sách các tài liệu/văn bản thuộc chủ đề khác nhau
# Mỗi tài liệu/văn bản sẽ tổ chức dạng 1 tuple với: (topic, nội_dung_văn_bản, danh_sách_token)
D = []

# Viết hàm tiền xử lý và tách từ tiếng Việt
def preprocess(doc):
  # Tiến hành xử lý các lỗi từ/câu, dấu câu, v.v. trong tiếng Việt với hàm text_normalize
  normalized_doc = text_normalize(doc)
  # Tiến hành tách từ
  tokens = word_tokenize(normalized_doc)
  # Tiến hành kết hợp các từ ghép trong tiếng Việt bằng '_'
  combined_tokens = [token.replace(' ', '_') for token in tokens]
  return (normalized_doc, combined_tokens)

# Viết hàm lấy danh sách các văn bản/tài liệu thuộc các chủ đề khác nhau
def fetch_doc_by_topic(topic):
  data_root_dir_path = 'data/vnexpress/{}'.format(topic)
  docs = []
  for file_name in os.listdir(data_root_dir_path):
    file_path = os.path.join(data_root_dir_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
      lines = []
      for line in f:
        line = line.lower().strip()
        lines.append(line)
    doc = " ".join(lines)
    clean_doc = re.sub('\W+',' ', doc)
    (normalized_doc, tokens) = preprocess(clean_doc)
    docs.append((topic, normalized_doc, tokens))
  return docs
topic_doc_idxes_dict = {}
doc_idx_topic_dict = {}

# Duyệt qua từng chủ đề
doc_idx = 0
for topic in topics:
  current_topic_docs = fetch_doc_by_topic(topic)
  topic_doc_idxes_dict[topic] = []
  for (topic, normalized_doc, tokens) in current_topic_docs:
    topic_doc_idxes_dict[topic].append(doc_idx)
    doc_idx_topic_dict[doc_idx] = topic
    doc_idx+=1
  D += current_topic_docs

doc_size = len(D)

print('Hoàn tất, tổng số lượng tài liệu/văn bản đã lấy: [{}]'.format(doc_size))
for topic in topic_doc_idxes_dict.keys():
  print(' - Chủ đề [{}] có [{}] tài liệu/văn bản.'.format(topic, len(topic_doc_idxes_dict[topic])))

# Khởi tạo đối tượng TfidfVectorizer
vectorizer = TfidfVectorizer()

# Chúng ta sẽ tạo ra một tập danh sách các tài liệu/văn bản dạng list đơn giản để thư viện Scikit-Learn có thể đọc được
sk_docs = []

# Duyệt qua từng tài liệu/văn bản có trong (D)
for (topic, normalized_doc, tokens) in D:
  # Chúng ta sẽ nối các từ/tokens đã được tách để làm thành một văn bản hoàn chỉnh
  text = ' '.join(tokens)
  sk_docs.append(text)

# Tiến hành chuyển đổi các tài liệu/văn bản về dạng các TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(sk_docs)

# Chuyển ma trận tfidf_matrix từ dạng cấu trúc thưa sang dạng đầy đủ để thuận tiện cho việc tính toán
tfidf_matrix = tfidf_matrix.todense()

# Viết hàm giúp chuyển đổi truy vấn dạng text sang tfidf vector
def parse_query(query_text):
  (normalized_doc, combined_tokens) = preprocess(query_text)
  query_text = ' '.join(combined_tokens)
  # Convert the query vector to a 1D array
  query_tfidf_vector = np.asarray(vectorizer.transform([query_text])[0].todense()).squeeze()
  return query_tfidf_vector

# Viết hàm giúp tìm kiếm top-k (mặc định 10) các kết quả tài liệu/văn bản tương đồng với truy vấn
def search(query_tfidf_vector, top_k = 10):
  search_results = {}
  for doc_idx, doc_tfidf_vector in enumerate(tfidf_matrix):
      # Convert the document vector to a 1D array
      doc_tfidf_vector = np.asarray(doc_tfidf_vector).squeeze()
      # Tính mức độ tương đồng giữa truy vấn (q) và từng tài liệu/văn bản (doc_idx) bằng độ đo cosine
      cs_score = 1 - distance.cosine(query_tfidf_vector, doc_tfidf_vector)
      search_results[doc_idx] = cs_score
  # Tiến hành sắp xếp các tài liệu/văn bản theo mức độ tương đồng từ cao -> thấp
  sorted_search_results = sorted(search_results.items(), key=lambda item: item[1], reverse=True)
  print('Top-[{}] tài liệu/văn bản có liên quan đến truy vấn.'.format(top_k))
  for idx, (doc_idx, sim_score) in enumerate(sorted_search_results[:top_k]):
    print(' - [{}]. Tài liệu [{}], chủ đề: [{}] -> mức độ tương đồng: [{:.6f}]'.format(idx + 1, doc_idx, doc_idx_topic_dict[doc_idx], sim_score))
  return sorted_search_results
# Thử một truy vấn với chủ đề [the-thao]
query_text = 'bóng đá'

# Chuyển đổi truy vấn về dạng vector tfidf
query_tfidf_vector = parse_query(query_text)

# Chúng ta sẽ thử tìm kiếm với top-10 kết quả
top_k = 10

# Tiến hành tìm kiếm thử trong tập dữ liệu với truy vấn
sorted_search_results = search(query_tfidf_vector, top_k)

# Chúng ta chọn mộ số lượng lượng top-m tài liệu/văn bản để xác định tập [D_pseudo_rel]
m = 5
D_pseudo_rel = []
top_m_results = sorted_search_results[:m]
for doc_idx, sim_score in top_m_results:
  D_pseudo_rel.append(doc_idx)

# Tạo mới một query mở rộng [query_tfidf_vector_QR] cho [query_tfidf_vector]
query_tfidf_vector_QR = query_tfidf_vector

# Với tập (D_pseudo_rel) tiến hành cộng các tfidf vector của các tài liệu/văn bản trong đó với query_tfidf_vector_QR
for doc_idx in D_pseudo_rel:
  doc_tfidf_vector_rel = np.asarray(tfidf_matrix[doc_idx]).squeeze()
  query_tfidf_vector_QR += doc_tfidf_vector_rel

# Tiến hành tìm kiếm lại với truy vấn mới[query_tfidf_vector_QR]
sorted_search_results = search(query_tfidf_vector_QR, top_k)