import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from underthesea import text_normalize
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance


from underthesea import ner
docOriginArr = []
docRankArr = []
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
    docRankArr.append((0,file_path))
    with open(file_path, 'r', encoding='utf-8') as f:
      lines = []
      for line in f:
        line = line.lower().strip()
        lines.append(line)
    doc = " ".join(lines)
    docOriginArr.append(doc)
    clean_doc = re.sub('\W+',' ', doc)
    (normalized_doc, tokens) = preprocess(clean_doc)
    docs.append((topic, normalized_doc, tokens))
  return docs

topics = [
    'du-lich',
    'suc-khoe',
    'the-gioi',
    'the-thao',
    'giao-duc',
    'khoa-hoc',
    'oto-xe-may'
]
docArr = []
for topic in topics:
  current_topic_docs = fetch_doc_by_topic(topic)
  docArr.append(current_topic_docs)

docs = docOriginArr

# Truy vấn (q)
query = 'du lịch bằng xe máy'

# Để cho tiện lợi trong việc xử lý chúng ta sẽ gán câu truy vấn là 1 tài liệu cuối cùng
docs.append(query)

doc_len = len(docs)

# Khởi tạo đối tượng TfidfVectorizer
vectorizer = TfidfVectorizer()

# Tiến hành chuyển đổi các tài liệu/văn bản và truy vấn về dạng các vector TF-IDF
# [tfidf_matrix] là một ma trận ở dạng thưa (sparse) - chỉ lưu các vị trí có giá trị khác 0  - chứa trọng số TF-IDF của các tài liệu/văn bản
tfidf_matrix = vectorizer.fit_transform(docs)

# Chúng ta tiến hành chuyển ma trận tfidf_matrix về dạng đầy đủ
tfidf_matrix = tfidf_matrix.todense()

# Lấy danh sách tập từ vựng
vocab = vectorizer.get_feature_names_out()
vocab_size = len(vocab)
print('Kích thước tập từ vựng: [{}]'.format(vocab_size))
print('Tập từ vựng (V):')
print(vocab)

# Chuyển đổi  ma trận (numpy) về dạng list
tfidf_matrix_list = tfidf_matrix.tolist()

# TFIDF encode của truy vấn (q) là tài liệu cuối cùng
# Kết quả tính TF-IDF của thư viện Scikit-Learn sẽ hơi khác với cách tính truyền thống
# vì IDF của Scikit-Learn sẽ là: idf(t) = loge [ (1+n) / ( 1 + df ) ] + 1
# sau đó toàn bộ ma trận TF-IDF sẽ được bình thường hoá lại với (norm - L2)
# tuy nhiên kết quả cuối cùng cũng sẽ không thay đổi
query_tfidf_encoded_vector = tfidf_matrix_list[doc_len-1]
print(query_tfidf_encoded_vector)

# Xóa query đã được mã hóa thành dạng tfidf vector ra khỏi tfidf_matrix_list
del tfidf_matrix_list[doc_len-1]

# Duyệt qua danh sách các tài liệu/văn bản (đã mã hóa ở dạng  vectors)

for doc_idx, doc_tfidf_encoded_vector in enumerate(tfidf_matrix_list):
    # Tính tích vô hướng giữa hai vectors tài liệu và truy vấn
    dot_product_sim = np.dot(query_tfidf_encoded_vector, doc_tfidf_encoded_vector)

    # Tính khoảng cách Euclid giữa hai vectors tài liệu và truy vấn
    ed = distance.euclidean(query_tfidf_encoded_vector, doc_tfidf_encoded_vector)

    # Tính tương đồ cosine giữa hai vectors tài liệu và truy vấn
    cs = 1 - distance.cosine(query_tfidf_encoded_vector, doc_tfidf_encoded_vector)
    docRankArr[doc_idx] = (cs, docRankArr[doc_idx][1])
    print('Tài liệu: [{}], tương đồng (dot product): [{:.6f}]'.format(doc_idx, dot_product_sim))
    print('Tài liệu: [{}], tương đồng (khoảng cách Euclid): [{:.6f}]'.format(doc_idx, ed))
    print('Tài liệu: [{}], tương đồng (Tương đồng cosine): [{:.6f}]'.format(doc_idx, cs))
    print('---')
docRankArr = sorted(docRankArr, key=lambda tup: tup[0], reverse=True)
for doc_idx, docRank in enumerate(docRankArr):
    print('[{}] {} {}'.format(doc_idx, docRank[0], docRank[1]))