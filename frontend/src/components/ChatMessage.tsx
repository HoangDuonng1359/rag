import React from 'react';
import { Message } from '../services/api';

interface ChatMessageProps {
  message: Message;
  isLatest?: boolean;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, isLatest }) => {
  return (
    <div className="space-y-4">
      {/* User Question */}
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-md">
          <p className="text-sm whitespace-pre-wrap break-words">{message.question}</p>
          <span className="text-xs text-blue-200 mt-1 block">
            {new Date(message.timestamp).toLocaleTimeString('vi-VN', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
        </div>
      </div>

      {/* AI Answer */}
      <div className="flex justify-start">
        <div className="max-w-[80%] bg-gray-100 text-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 shadow-md">
          <div className="flex items-start space-x-2">
            <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 2a8 8 0 100 16 8 8 0 000-16zM9 9a1 1 0 012 0v4a1 1 0 11-2 0V9zm1-4a1 1 0 100 2 1 1 0 000-2z" />
              </svg>
            </div>
            <div className="flex-1">
              <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">
                {message.answer}
              </p>
              <span className="text-xs text-gray-500 mt-1 block">
                GraphRAG • {new Date(message.timestamp).toLocaleTimeString('vi-VN', {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
