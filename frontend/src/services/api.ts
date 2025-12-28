// API Service for GraphRAG Backend

const API_BASE_URL = 'http://localhost:8000';

export interface Message {
  question: string;
  answer: string;
  timestamp: string;
  model?: 'graphrag' | 'rag';
}

export interface SessionResponse {
  session_id: string;
  created_at: string;
  message: string;
}

export interface GraphRAGResponse {
  session_id: string;
  question: string;
  answer: string;
  metadata: {
    retrieval_time: number;
    generation_time: number;
    total_time: number;
    entities_used: number;
    relationships_used: number;
  };
}

export interface SessionHistory {
  session_id: string;
  created_at: string;
  history: Message[];
}

// Create new session
export const createSession = async (): Promise<SessionResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/create_new_session`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  
  if (!response.ok) {
    throw new Error('Failed to create session');
  }
  
  return response.json();
};

// Ask question
export const askQuestion = async (
  sessionId: string | null,
  question: string
): Promise<GraphRAGResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/graphrag`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      question: question,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to get answer');
  }
  
  return response.json();
};

// Ask question using Traditional RAG
export const askQuestionRAG = async (
  sessionId: string | null,
  question: string
): Promise<GraphRAGResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/rag`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      question: question,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to get answer');
  }
  
  return response.json();
};

// Get session history
export const getSessionHistory = async (
  sessionId: string
): Promise<SessionHistory> => {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}`);
  
  if (!response.ok) {
    throw new Error('Failed to get session history');
  }
  
  return response.json();
};

// Health check
export const healthCheck = async (): Promise<any> => {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  
  if (!response.ok) {
    throw new Error('Backend not healthy');
  }
  
  return response.json();
};
