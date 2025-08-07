from array import ArrayType
import os
import re
import math
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from wordcloud import WordCloud
from underthesea import text_normalize
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
import json

class Query:

  # Tạo một tập dữ liệu thử nghiệm gồm các tài liệu/văn bản thuộc về 2-3 chủ đề
  # Cấu trúc dữ liệu dạng list - lưu thông tin danh sách các tài liệu/văn bản thuộc chủ đề khác nhau
  # Mỗi tài liệu/văn bản sẽ tổ chức dạng 1 tuple với: (topic, nội_dung_văn_bản, danh_sách_token)
  D = []
  DocOriginArr = []
  currentTerm = ""
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
    data_root_dir_path = '../data/vnexpress/{}'.format(topic)
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
      Query.DocOriginArr.append(clean_doc)
      (normalized_doc, tokens) = Query.preprocess(clean_doc)
      docs.append((topic, normalized_doc, tokens))
    return docs
  
  topic_doc_idxes_dict = {}
  doc_idx_topic_dict = {}
  tfidf_matrix:ArrayType #2d
  doc_tfidf_vector_rel:ArrayType #2d
  vectorizer = TfidfVectorizer()
  # Viết hàm giúp chuyển đổi truy vấn dạng text sang tfidf vector
  def parse_query(query_text):
    (normalized_doc, combined_tokens) = Query.preprocess(query_text)
    query_text = ' '.join(combined_tokens)
    # Convert the query vector to a 1D array
    query_tfidf_vector = np.asarray(Query.vectorizer.transform([query_text])[0].todense()).squeeze()
    return query_tfidf_vector

  # Viết hàm giúp tìm kiếm top-k (mặc định 10) các kết quả tài liệu/văn bản tương đồng với truy vấn
  def search(query_tfidf_vector, top_k = 10):
    search_results = {}
    for doc_idx, doc_tfidf_vector in enumerate(Query.tfidf_matrix):
        # Convert the document vector to a 1D array
        doc_tfidf_vector = np.asarray(doc_tfidf_vector).squeeze()
        # Tính mức độ tương đồng giữa truy vấn (q) và từng tài liệu/văn bản (doc_idx) bằng độ đo cosine
        cs_score = 1 - distance.cosine(query_tfidf_vector, doc_tfidf_vector)
        search_results[doc_idx] = cs_score
    # Tiến hành sắp xếp các tài liệu/văn bản theo mức độ tương đồng từ cao -> thấp
    sorted_search_results = sorted(search_results.items(), key=lambda item: item[1], reverse=True)
    print('Top-[{}] tài liệu/văn bản có liên quan đến truy vấn.'.format(top_k))
    for idx, (doc_idx, sim_score) in enumerate(sorted_search_results[:top_k]):
      print(' - [{}]. Tài liệu [{}], chủ đề: [{}] -> mức độ tương đồng: [{:.6f}]'.format(idx + 1, doc_idx, Query.doc_idx_topic_dict[doc_idx], sim_score))
    return sorted_search_results[:top_k]

  def doSearch(queryText:str, docRelIds:ArrayType, top_k:int = 10) -> ArrayType:
    if(queryText != Query.currentTerm):
      Query.doc_tfidf_vector_rel = []
      docRelIds = []
    Query.currentTerm = queryText
    query_tfidf_vector = Query.parse_query(queryText)
    for doc_idx in docRelIds:
      if len(Query.doc_tfidf_vector_rel) == 0:
        Query.doc_tfidf_vector_rel = np.asarray(Query.tfidf_matrix[doc_idx]).squeeze()
      else:
        Query.doc_tfidf_vector_rel += np.asarray(Query.tfidf_matrix[doc_idx]).squeeze()
      query_tfidf_vector += Query.doc_tfidf_vector_rel
    results = Query.search(query_tfidf_vector, top_k)
    return [{"id":result[0],"content": Query.DocOriginArr[result[0]]} for result in results]
  
  def init():
    topics = [
      'the-thao',
      'giao-duc',
      'khoa-hoc'
    ]
    # Duyệt qua từng chủ đề
    Query.D = []
    Query.DocOriginArr = []
    doc_idx = 0
    for topic in topics:
      current_topic_docs = Query.fetch_doc_by_topic(topic)
      Query.topic_doc_idxes_dict[topic] = []
      for (topic, normalized_doc, tokens) in current_topic_docs:
        Query.topic_doc_idxes_dict[topic].append(doc_idx)
        Query.doc_idx_topic_dict[doc_idx] = topic
        doc_idx+=1
      Query.D += current_topic_docs

      doc_size = len(Query.D)

      print('Hoàn tất, tổng số lượng tài liệu/văn bản đã lấy: [{}]'.format(doc_size))
      for topic in Query.topic_doc_idxes_dict.keys():
        print(' - Chủ đề [{}] có [{}] tài liệu/văn bản.'.format(topic, len(Query.topic_doc_idxes_dict[topic])))

      # Chúng ta sẽ tạo ra một tập danh sách các tài liệu/văn bản dạng list đơn giản để thư viện Scikit-Learn có thể đọc được
      sk_docs = []

      # Duyệt qua từng tài liệu/văn bản có trong (D)
      for (topic, normalized_doc, tokens) in Query.D:
        # Chúng ta sẽ nối các từ/tokens đã được tách để làm thành một văn bản hoàn chỉnh
        text = ' '.join(tokens)
        sk_docs.append(text)

      # Tiến hành chuyển đổi các tài liệu/văn bản về dạng các TF-IDF vectors
      Query.tfidf_matrix = Query.vectorizer.fit_transform(sk_docs)

      # Chuyển ma trận tfidf_matrix từ dạng cấu trúc thưa sang dạng đầy đủ để thuận tiện cho việc tính toán
      Query.tfidf_matrix = Query.tfidf_matrix.todense()

'''
Query.init()
results = Query.doSearch('bóng đá',[],10)
for result in enumerate(results):
  print(result)
print("Doc relevance")
docR = [r[0] for r in results[:5]]
print(docR)
results = Query.doSearch('bóng đá',docR,10)
'''

from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask.json import jsonify

app = Flask(__name__)
CORS(app)
api = Api(app)
Query.init()

@app.route('/search', methods=['POST'])
def search():
  if request.is_json:
      data = request.get_json()
      # Process the received JSON data
      print(f"Received data: {data}")
      print(f"data: {data['term']}")
      result = Query.doSearch(data['term'], data['rdoc'], data['limit'])
      return jsonify(result), 200
  else:
      return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
  app.run()