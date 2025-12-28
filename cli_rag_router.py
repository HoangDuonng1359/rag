import sys
import argparse
from router import create_router_with_defaults


def main():
    parser = argparse.ArgumentParser(
        description="RAG Router - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""""
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Câu hỏi cần trả lời (nếu không có sẽ chạy interactive mode)'
    )
    
    parser.add_argument(
        '--force',
        type=str,
        choices=['factual', 'relational'],
        help='Force sử dụng một hệ thống cụ thể (factual=Traditional RAG, relational=Graph RAG)'
    )
    
    parser.add_argument(
        '--no-classify',
        action='store_true',
        help='Không tự động phân loại (mặc định dùng Traditional RAG)'
    )
    
    parser.add_argument(
        '--chroma-path',
        type=str,
        default='chroma_db',
        help='Đường dẫn đến ChromaDB (mặc định: chroma_db)'
    )
    
    args = parser.parse_args()
    
    # Khởi tạo router
    print("Đang khởi tạo Intelligent RAG Router...")
    try:
        router = create_router_with_defaults(chroma_path=args.chroma_path)
    except Exception as e:
        print(f"Lỗi khi khởi tạo router: {e}")
        return 1
    
    # Nếu có query, xử lý và thoát
    if args.query:
        try:
            from router import QueryType
            
            # Xác định force_type
            force_type = None
            if args.force:
                force_type = QueryType.FACTUAL if args.force == 'factual' else QueryType.RELATIONAL
            
            # Query
            result = router.route_query(
                question=args.query,
                auto_classify=not args.no_classify,
                force_type=force_type
            )
            
            # Hiển thị kết quả
            print(f"Câu hỏi: {result['question']}")
            print(f"Hệ thống: {result['system_used'].upper()}")
            print(f"Loại: {result['query_type']}")
            print(f"\nCâu trả lời:\n")
            print(result['answer'])
            
            return 0
            
        except Exception as e:
            print(f"Lỗi khi xử lý câu hỏi: {e}")
            return 1
    else:
        try:
            router.interactive_query()
            return 0
        except Exception as e:
            print(f"Lỗi: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
