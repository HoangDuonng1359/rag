import re
import json
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridChunker:
    def __init__(self, max_words=500, min_words=300, overlap_chars=200):
        self.max_words = max_words
        self.min_words = min_words
        self.overlap_chars = overlap_chars
        
    def count_words(self, text):
        """Đếm số từ trong văn bản tiếng Việt"""
        return len(text.split())
    
    def split_sentences(self, text):
        """Tách văn bản thành các câu"""
        # Tách theo dấu chấm câu lớn, giữ nguyên dấu
        sentences = re.split(r'([.!?]\s+)', text)
        # Ghép lại dấu với câu
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])
        return [s.strip() for s in result if s.strip()]
    
    def find_semantic_breakpoints(self, sentences, target_words):
        """Tìm điểm chia tốt nhất dựa trên semantic similarity"""
        if len(sentences) <= 2:
            return []
        
        # Tính TF-IDF cho các câu
        try:
            vectorizer = TfidfVectorizer(max_features=100)
            vectors = vectorizer.fit_transform(sentences)
            
            # Tính similarity giữa các câu liên tiếp
            similarities = []
            for i in range(len(sentences)-1):
                sim = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
                similarities.append(sim)
            
            # Tìm vị trí có similarity thấp (topic thay đổi)
            # và gần với target_words nhất
            word_counts = [self.count_words(s) for s in sentences]
            cumsum = np.cumsum(word_counts)
            
            breakpoints = []
            for i, sim in enumerate(similarities):
                # Ưu tiên điểm có similarity thấp và gần target
                if cumsum[i] >= self.min_words and cumsum[i] <= self.max_words * 1.2:
                    score = (1 - sim) * 100 - abs(cumsum[i] - target_words)
                    breakpoints.append((i+1, score))
            
            # Chọn điểm tốt nhất
            if breakpoints:
                breakpoints.sort(key=lambda x: x[1], reverse=True)
                return [breakpoints[0][0]]
            
        except Exception as e:
            print(f"Warning: Semantic analysis failed: {e}")
        
        return []
    
    def parse_markdown_structure(self, content):
        """Parse markdown theo cấu trúc hierarchical"""
        sections = []
        current_section = {
            'header': None,
            'chapter': None,
            'content': [],
            'page_start': None,
            'page_end': None
        }
        
        lines = content.split('\n')
        in_footer = False
        in_header = False
        current_page = None
        
        for line in lines:
            # Phát hiện page marker
            if line.startswith('--- Page '):
                match = re.search(r'--- Page (\d+) ---', line)
                if match:
                    current_page = int(match.group(1))
                    if current_section['page_start'] is None:
                        current_section['page_start'] = current_page
                    current_section['page_end'] = current_page
                continue
            
            # Bỏ qua footer
            if line.strip() == '<footer>':
                in_footer = True
                continue
            if line.strip() == '</footer>':
                in_footer = False
                continue
            if in_footer:
                continue
            
            # Phát hiện header (tiêu đề chương/mục)
            if line.strip() == '<header>':
                in_header = True
                continue
            if line.strip() == '</header>':
                in_header = False
                # Lưu section cũ nếu có nội dung
                if current_section['content']:
                    sections.append(current_section)
                # Bắt đầu section mới
                current_section = {
                    'header': current_section['header'],
                    'chapter': current_section['chapter'],
                    'content': [],
                    'page_start': current_page,
                    'page_end': current_page
                }
                continue
            
            if in_header:
                current_section['header'] = line.strip()
                # Phát hiện chương
                if re.match(r'Chương [IVX]+', line) or re.match(r'CHƯƠNG [IVX]+', line):
                    current_section['chapter'] = line.strip()
                continue
            
            # Bỏ qua table markers (giữ nguyên trong content)
            if '**[TABLE' in line or '**[END OF TABLE' in line:
                continue
            
            # Thêm nội dung
            if line.strip():
                current_section['content'].append(line)
        
        # Lưu section cuối
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def chunk_section(self, section):
        """Chia một section thành chunks"""
        content_text = '\n'.join(section['content'])
        word_count = self.count_words(content_text)
        
        chunks = []
        
        # Nếu section nhỏ, giữ nguyên
        if word_count <= self.max_words:
            chunks.append({
                'content': content_text,
                'metadata': {
                    'chapter': section['chapter'],
                    'header': section['header'],
                    'page_start': section['page_start'],
                    'page_end': section['page_end'],
                    'word_count': word_count
                }
            })
        else:
            # Section lớn, chia nhỏ
            # Bước 1: Tách thành câu
            sentences = self.split_sentences(content_text)
            
            # Bước 2: Tìm breakpoints semantic
            current_chunk = []
            current_words = 0
            
            for i, sentence in enumerate(sentences):
                sentence_words = self.count_words(sentence)
                
                # Kiểm tra xem có nên bắt đầu chunk mới không
                if current_words + sentence_words > self.max_words and current_words >= self.min_words:
                    # Lưu chunk hiện tại
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'chapter': section['chapter'],
                            'header': section['header'],
                            'page_start': section['page_start'],
                            'page_end': section['page_end'],
                            'word_count': current_words
                        }
                    })
                    
                    # Bắt đầu chunk mới với overlap
                    overlap_text = chunk_text[-self.overlap_chars:] if len(chunk_text) > self.overlap_chars else chunk_text
                    current_chunk = [overlap_text, sentence]
                    current_words = self.count_words(overlap_text) + sentence_words
                else:
                    current_chunk.append(sentence)
                    current_words += sentence_words
            
            # Lưu chunk cuối
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'chapter': section['chapter'],
                        'header': section['header'],
                        'page_start': section['page_start'],
                        'page_end': section['page_end'],
                        'word_count': self.count_words(chunk_text)
                    }
                })
        
        return chunks
    
    def process_file(self, input_path, output_path):
        """Xử lý file markdown và lưu chunks"""
        print(f"Đang đọc file: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("Đang parse cấu trúc văn bản...")
        sections = self.parse_markdown_structure(content)
        print(f"Tìm thấy {len(sections)} sections")
        
        print("Đang tạo chunks...")
        all_chunks = []
        for i, section in enumerate(sections):
            chunks = self.chunk_section(section)
            for chunk in chunks:
                chunk['id'] = f"chunk_{len(all_chunks)+1:04d}"
                all_chunks.append(chunk)
        
        print(f"Đã tạo {len(all_chunks)} chunks")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Lưu vào JSON
        output_data = {
            'source_file': input_path,
            'total_chunks': len(all_chunks),
            'config': {
                'max_words': self.max_words,
                'min_words': self.min_words,
                'overlap_chars': self.overlap_chars
            },
            'chunks': all_chunks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu vào: {output_path}")
        
        # In thống kê
        word_counts = [c['metadata']['word_count'] for c in all_chunks]
        print(f"\nThống kê:")
        print(f"  - Số chunks: {len(all_chunks)}")
        print(f"  - Trung bình: {np.mean(word_counts):.1f} từ/chunk")
        print(f"  - Min: {min(word_counts)} từ")
        print(f"  - Max: {max(word_counts)} từ")


if __name__ == "__main__":
    # Cấu hình
    input_file = "./data/chapter10.md"
    output_file = "./data/chunk/chapter10_chunk.json"
    
    # Tạo chunker và xử lý
    chunker = HybridChunker(max_words=500, min_words=300, overlap_chars=200)
    chunker.process_file(input_file, output_file)
    
    print("\nHoàn tất!")