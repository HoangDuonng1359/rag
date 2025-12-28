"""
Debug rerank model
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading rerank model...")
tokenizer = AutoTokenizer.from_pretrained(
    "huynhdat543/VietNamese_law_rerank",
    trust_remote_code=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "huynhdat543/VietNamese_law_rerank",
    trust_remote_code=True,
).to("cpu")

query = "Xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?"
passage = """Điều 7. Xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ
5. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng, trừ 02 điểm giấy phép lái xe đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;"""

text = f"[Q] {query}\n[P] {passage}"

print(f"\nInput text:\n{text[:200]}...")
print(f"\nInput length: {len(text)}")

enc = tokenizer(
    [text],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

print(f"\nTokenized shape: {enc['input_ids'].shape}")
print(f"First 20 tokens: {enc['input_ids'][0][:20]}")

with torch.no_grad():
    outputs = model(**enc)
    logits = outputs.logits
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits: {logits}")
    
    if logits.shape[1] == 2:
        scores = logits[:, 1]
        print(f"\nBinary classification - using logit[:, 1]")
        print(f"Score (logit class 1): {scores}")
        
        # Thử softmax
        probs = torch.softmax(logits, dim=1)
        print(f"Softmax probabilities: {probs}")
        print(f"Probability of relevant (class 1): {probs[0][1].item()}")
    else:
        scores = logits.squeeze(-1)
        print(f"\nSingle score: {scores}")

print("\n" + "="*80)
print("Testing with irrelevant passage")
print("="*80)

irrelevant = "Hôm nay trời đẹp và tôi đi mua sắm ở chợ"
text2 = f"[Q] {query}\n[P] {irrelevant}"

enc2 = tokenizer(
    [text2],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

with torch.no_grad():
    outputs2 = model(**enc2)
    logits2 = outputs2.logits
    
    print(f"\nLogits: {logits2}")
    
    if logits2.shape[1] == 2:
        probs2 = torch.softmax(logits2, dim=1)
        print(f"Softmax probabilities: {probs2}")
        print(f"Probability of relevant (class 1): {probs2[0][1].item()}")
