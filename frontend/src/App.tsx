import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { ConversationList } from './components/ConversationList';
import * as api from './services/api';
import type { Message } from './services/api';

interface ChatMetadata {
  totalTime?: number;
  entitiesUsed?: number;
  relationshipsUsed?: number;
}

interface Conversation {
  id: string;
  title: string;
  lastMessage?: string;
  timestamp: string;
  messageCount: number;
  messages: Message[];
  metadata: ChatMetadata | null;
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'graphrag' | 'rag'>('graphrag');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Get current conversation data
  const currentConversation = conversations.find(c => c.id === currentConversationId);
  const messages = currentConversation?.messages || [];
  const lastMetadata = currentConversation?.metadata || null;

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const checkBackendHealth = async () => {
    try {
      await api.healthCheck();
      setIsBackendHealthy(true);
    } catch (error) {
      console.error('Backend health check failed:', error);
      setIsBackendHealthy(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const createNewSession = async () => {
    try {
      const response = await api.createSession();
      const newConversation: Conversation = {
        id: response.session_id,
        title: 'Hội thoại mới',
        timestamp: new Date().toISOString(),
        messageCount: 0,
        messages: [],
        metadata: null,
      };
      
      setConversations(prev => [newConversation, ...prev]);
      setCurrentConversationId(response.session_id);
      console.log('New session created:', response.session_id);
    } catch (error) {
      console.error('Failed to create session:', error);
      alert('Không thể tạo session mới. Vui lòng kiểm tra kết nối backend.');
    }
  };

  const selectConversation = (id: string) => {
    setCurrentConversationId(id);
  };

  const handleSendMessage = async (question: string) => {
    if (!question.trim()) return;

    // Create new conversation if none exists
    let targetConvId = currentConversationId;
    if (!targetConvId) {
      try {
        const response = await api.createSession();
        const newConversation: Conversation = {
          id: response.session_id,
          title: question.substring(0, 50) + (question.length > 50 ? '...' : ''),
          timestamp: new Date().toISOString(),
          messageCount: 0,
          messages: [],
          metadata: null,
        };
        
        setConversations(prev => [newConversation, ...prev]);
        setCurrentConversationId(response.session_id);
        targetConvId = response.session_id;
      } catch (error) {
        console.error('Failed to create session:', error);
        alert('Không thể tạo session mới.');
        return;
      }
    }

    setIsLoading(true);

    try {
      // Call appropriate API based on selected model
      const response = selectedModel === 'graphrag' 
        ? await api.askQuestion(targetConvId, question)
        : await api.askQuestionRAG(targetConvId, question);

      // Add new message to history
      const newMessage: Message = {
        question: response.question,
        answer: response.answer,
        timestamp: new Date().toISOString(),
      };

      const newMetadata: ChatMetadata = {
        totalTime: response.metadata.total_time,
        entitiesUsed: response.metadata.entities_used,
        relationshipsUsed: response.metadata.relationships_used,
      };

      // Update conversation
      setConversations(prev => prev.map(conv => {
        if (conv.id === targetConvId) {
          const updatedMessages = [...conv.messages, newMessage];
          return {
            ...conv,
            messages: updatedMessages,
            messageCount: updatedMessages.length,
            lastMessage: question,
            timestamp: new Date().toISOString(),
            metadata: newMetadata,
            // Update title with first question if still default
            title: conv.title === 'Hội thoại mới' 
              ? question.substring(0, 50) + (question.length > 50 ? '...' : '')
              : conv.title,
          };
        }
        return conv;
      }));

    } catch (error) {
      console.error('Failed to send message:', error);
      alert('Không thể gửi tin nhắn. Vui lòng thử lại.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar - Conversation List */}
      <div className="w-80 flex-shrink-0">
        <ConversationList
          conversations={conversations}
          currentSessionId={currentConversationId}
          onSelectConversation={selectConversation}
          onCreateNew={createNewSession}
          isBackendHealthy={isBackendHealthy}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-gradient-to-br from-blue-50 to-indigo-100">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                    <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">
                    {currentConversation?.title || 'Graph RAG Chat'}
                  </h1>
                  <p className="text-sm text-gray-500">
                    Hỏi đáp về luật giao thông Việt Nam
                  </p>
                </div>
              </div>

              {currentConversationId && lastMetadata && (
                <div className="flex items-center space-x-4 text-gray-500 text-sm">
                  <div className="flex items-center space-x-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                    </svg>
                    <span>{lastMetadata.totalTime?.toFixed(2)}s</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                      <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                    </svg>
                    <span>{lastMetadata.entitiesUsed} entities</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M12.586 4.586a2 2 0 112.828 2.828l-3 3a2 2 0 01-2.828 0 1 1 0 00-1.414 1.414 4 4 0 005.656 0l3-3a4 4 0 00-5.656-5.656l-1.5 1.5a1 1 0 101.414 1.414l1.5-1.5zm-5 5a2 2 0 012.828 0 1 1 0 101.414-1.414 4 4 0 00-5.656 0l-3 3a4 4 0 105.656 5.656l1.5-1.5a1 1 0 10-1.414-1.414l-1.5 1.5a2 2 0 11-2.828-2.828l3-3z" clipRule="evenodd" />
                    </svg>
                    <span>{lastMetadata.relationshipsUsed} links</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Chat Messages Area */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-6 py-6">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center py-20">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                    <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">
                  Chào mừng đến với Graph RAG Chat!
                </h2>

                {/* Example questions */}
                <div className="mt-10 max-w-2xl">
                  <p className="text-sm text-gray-500 mb-4">Câu hỏi mẫu:</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {[
                      'Mức phạt vượt đèn đỏ đối với xe máy?',
                      'Xe máy có bắt buộc đội mũ bảo hiểm không?',
                      'Tốc độ tối đa trong khu dân cư?',
                      'Quy định về nồng độ cồn khi lái xe?',
                    ].map((q, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSendMessage(q)}
                        disabled={!isBackendHealthy || isLoading}
                        className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors duration-200 text-left text-sm text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((message, index) => (
                  <ChatMessage 
                    key={index} 
                    message={message}
                    isLatest={index === messages.length - 1}
                  />
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 rounded-2xl px-4 py-3">
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                        <span className="text-sm text-gray-600">Đang suy nghĩ...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </main>

        {/* Input Area */}
        <ChatInput 
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          disabled={!isBackendHealthy}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
        />
      </div>
    </div>
  );
}

export default App;