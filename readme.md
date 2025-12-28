# Hệ thống chatbot luật giao thông
Mô tả hệ thống:
Hệ thống này sử dụng 2 kỹ thuật chính: RAG (Retrieval-Augmented Generation) và GRAPH - RAG để cung cấp câu trả lời chính xác và có căn cứ dựa trên các văn bản luật giao thông.


# Các thành viên nhóm:
- Hoàng Văn Dương
- Nguyễn Văn Huy
- Nguyễn Anh Kiệt

# Quick Start
1. Cài đặt môi trường ảo và cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
2. Thiết lập biến môi trường trong file `.env` với khóa API và thông tin kết nối cơ sở dữ liệu Neo4j.
3. Chạy hệ thống:
3.1. Chạy backend:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
3.2 . Chạy frontend:
```bash
cd frontend
npm install
npm start
```

Mở trình duyệt và truy cập `http://localhost:3000` để sử dụng hệ thống chatbot.

# Kỹ thuật chính:
- RAG (Retrieval-Augmented Generation): Kết hợp mô hình ngôn ngữ lớn với hệ thống truy xuất thông tin để cung cấp câu trả lời dựa trên các tài liệu luật giao thông.
- GRAPH - RAG: Sử dụng cơ sở dữ liệu đồ thị Neo4j để lưu trữ và truy xuất thông tin luật giao thông, kết hợp với RAG để nâng cao hiệu quả truy xuất và độ chính xác của câu trả lời.
# Các mô hình LLM sử dụng:
- Qwen-7B-Instruct
- Gemini 2.5 Flash