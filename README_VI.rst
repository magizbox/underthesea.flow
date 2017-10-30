================
Underthesea Flow
================

It's a very messy project, contains a lot of useful stuffs for NLP experiments

Flow
====

.. image:: https://raw.githubusercontent.com/magizbox/underthesea.flow/master/flow.png

Flow has **data readers**, **transformers**, **models**, **scores**, **validation methods**

Data Readers
============

Available data readers: `TaggedCorpus`

Transformers
============

Available transformers: `TaggedTransformer`

Models
======

Available models: Conditional Random Fields (`CRF`)

Validation
==========

Available validation methods: TrainTestSplit, CrossValidation

Available scores: `f1`, `accuracy`

Huấn luyện một model của riêng bạn
==================================

1. Chuẩn bị dữ liệu

Tạo thêm một thư mục mới trong thư mục data. Ví dụ `data1`

Bước 1: Thu thập dữ liệu

Lưu dữ liệu của bạn vào thư mục raw trong thư mục `data1`

Bước 2: Tiền xử lý và chuẩn hóa dữ liệu

Thay đổi script `preprocess.py` trong thư mục `data1` để chuẩn hóa và tiền xử lý dữ liệu của bạn.

Quá trình chuẩn hóa nên theo format chung, với bài toán phân loại như text classification, sentiment analysis (tham khảo classification_format.md). Với bài toán gán nhãn chuỗi như word segmentation, pos tagging, chunking, named entity recognition, dependency parsing (tham khảo conll_format.md)

Kết quả của quá trình tiền xử lý và chuẩn hóa dữ liệu sẽ cho ra một corpus sẵn sàng cho việc huấn luyện mô hình.

Bước 3: Phân tích thăm dò

Phân tích thăm dò là quá trình đưa ra các thống kê sơ bộ, các điểm đặc biệt trong corpus.

Thay đổi script `eda.py` trong thư mục `data1` để phân tích thăm dò corpus. Kết quả lưu vào thư mục `eda` tương ứng.




