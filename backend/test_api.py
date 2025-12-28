"""
Test script để demo GraphRAG API
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

def test_health():
    """Test health check"""
    print_section("1. HEALTH CHECK")
    response = requests.get(f"{BASE_URL}/api/health")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.status_code == 200

def test_create_session():
    """Test tạo session mới"""
    print_section("2. TẠO SESSION MỚI")
    response = requests.post(f"{BASE_URL}/api/create_new_session")
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data.get("session_id")

def test_ask_question(session_id, question):
    """Test hỏi câu hỏi"""
    print_section(f"3. HỎI CÂU HỎI")
    print(f"Question: {question}\n")
    
    payload = {
        "session_id": session_id,
        "question": question
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/graphrag", json=payload)
    elapsed = time.time() - start_time
    
    data = response.json()
    
    print(f"Session ID: {data.get('session_id')}")
    print(f"\nAnswer:")
    print(data.get('answer'))
    print(f"\nMetadata:")
    print(f"  - Total time: {data.get('metadata', {}).get('total_time', 0):.2f}s")
    print(f"  - Entities used: {data.get('metadata', {}).get('entities_used', 0)}")
    print(f"  - Relationships used: {data.get('metadata', {}).get('relationships_used', 0)}")
    print(f"  - API response time: {elapsed:.2f}s")
    
    return data.get("session_id")

def test_get_history(session_id):
    """Test lấy lịch sử"""
    print_section("4. LỊCH SỬ HỘI THOẠI")
    response = requests.get(f"{BASE_URL}/api/session/{session_id}")
    data = response.json()
    
    print(f"Session ID: {data.get('session_id')}")
    print(f"Created at: {data.get('created_at')}")
    print(f"Number of messages: {len(data.get('history', []))}")
    
    for i, msg in enumerate(data.get('history', []), 1):
        print(f"\n[{i}] {msg.get('timestamp')}")
        print(f"Q: {msg.get('question')}")
        print(f"A: {msg.get('answer')[:100]}...")

def main():
    print_section("GRAPH RAG API TEST")
    print("Base URL:", BASE_URL)
    
    # 1. Health check
    if not test_health():
        print("\n❌ Server not healthy!")
        return
    
    # 2. Tạo session
    session_id = test_create_session()
    if not session_id:
        print("\n❌ Failed to create session!")
        return
    
    # 3. Hỏi câu hỏi
    questions = [
        "Mức phạt vi phạm vượt đèn đỏ đối với xe máy là bao nhiêu?",
        "Xe máy phải đội mũ bảo hiểm không?",
    ]
    
    for question in questions:
        session_id = test_ask_question(session_id, question)
        time.sleep(1)  # Delay giữa các câu hỏi
    
    # 4. Xem lịch sử
    test_get_history(session_id)
    
    print_section("✓ TEST COMPLETED")

if __name__ == "__main__":
    main()
