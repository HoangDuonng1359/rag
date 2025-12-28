import React from 'react';

interface Conversation {
  id: string;
  title: string;
  lastMessage?: string;
  timestamp: string;
  messageCount: number;
}

interface ConversationListProps {
  conversations: Conversation[];
  currentSessionId: string | null;
  onSelectConversation: (id: string) => void;
  onCreateNew: () => void;
  isBackendHealthy: boolean;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  currentSessionId,
  onSelectConversation,
  onCreateNew,
  isBackendHealthy,
}) => {
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Vừa xong';
    if (minutes < 60) return `${minutes} phút trước`;
    if (hours < 24) return `${hours} giờ trước`;
    if (days < 7) return `${days} ngày trước`;
    return date.toLocaleDateString('vi-VN');
  };

  return (
    <div className="flex flex-col h-full bg-white border-r border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Hội thoại</h2>
          <div className={`w-2 h-2 rounded-full ${isBackendHealthy ? 'bg-green-500' : 'bg-red-500'}`} />
        </div>
        
        <button
          onClick={onCreateNew}
          disabled={!isBackendHealthy}
          className="w-full px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-2 shadow-md"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
          </svg>
          <span className="font-medium">Hội thoại mới</span>
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-3">
              <svg className="w-8 h-8 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
              </svg>
            </div>
            <p className="text-sm text-gray-500">
              Chưa có hội thoại nào.
              <br />
              Tạo mới để bắt đầu!
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => onSelectConversation(conv.id)}
                className={`w-full text-left p-4 hover:bg-gray-50 transition-colors duration-150 ${
                  currentSessionId === conv.id ? 'bg-blue-50 border-r-4 border-blue-600' : ''
                }`}
              >
                <div className="flex items-start justify-between mb-1">
                  <h3 className="font-medium text-gray-900 truncate flex-1 pr-2">
                    {conv.title}
                  </h3>
                  <span className="text-xs text-gray-500 whitespace-nowrap">
                    {formatTime(conv.timestamp)}
                  </span>
                </div>
                
                {conv.lastMessage && (
                  <p className="text-sm text-gray-600 truncate mb-1">
                    {conv.lastMessage}
                  </p>
                )}
                
                <div className="flex items-center space-x-2">
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                    {conv.messageCount} tin nhắn
                  </span>
                  <span className="text-xs text-gray-400">
                    ID: {conv.id.substring(0, 8)}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <div className="flex items-center space-x-2 text-xs text-gray-500">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <span>{conversations.length} hội thoại</span>
        </div>
      </div>
    </div>
  );
};
