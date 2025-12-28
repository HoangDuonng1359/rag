import React, { useState } from 'react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
  selectedModel: 'graphrag' | 'rag';
  onModelChange: (model: 'graphrag' | 'rag') => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  isLoading,
  disabled = false,
  selectedModel,
  onModelChange
}) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-gray-200 bg-white p-4">
      <div className="flex items-end space-x-2">
        <div className="flex-1 relative">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={disabled ? "Tạo session mới để bắt đầu..." : "Nhập câu hỏi của bạn..."}
            disabled={disabled || isLoading}
            rows={1}
            className="w-full px-4 py-3 pr-32 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none disabled:bg-gray-100 disabled:cursor-not-allowed"
            style={{ minHeight: '48px', maxHeight: '120px' }}
          />
          
          {/* Model Selector Dropdown */}
          <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
            <select
              value={selectedModel}
              onChange={(e) => onModelChange(e.target.value as 'graphrag' | 'rag')}
              disabled={disabled || isLoading}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-md bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed cursor-pointer"
            >
              <option value="graphrag">Graph RAG</option>
              <option value="rag">Traditional RAG</option>
            </select>
          </div>
        </div>
        
        <button
          type="submit"
          disabled={!input.trim() || isLoading || disabled}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200 flex items-center space-x-2"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle 
                  className="opacity-25" 
                  cx="12" 
                  cy="12" 
                  r="10" 
                  stroke="currentColor" 
                  strokeWidth="4"
                  fill="none"
                />
                <path 
                  className="opacity-75" 
                  fill="currentColor" 
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span>Đang xử lý...</span>
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
              <span>Gửi</span>
            </>
          )}
        </button>
      </div>
      <div className="flex justify-between items-center mt-2">
        <p className="text-xs text-gray-500">
          Nhấn Enter để gửi, Shift + Enter để xuống dòng
        </p>
        <p className="text-xs text-gray-600">
          Model: <span className="font-semibold">{selectedModel === 'graphrag' ? 'Graph RAG' : 'Traditional RAG'}</span>
        </p>
      </div>
    </form>
  );
};
