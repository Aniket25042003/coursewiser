/**
 * API service for backend communication
 */
import axios, { AxiosInstance } from 'axios';
import { getCurrentUserToken } from './firebase';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  async (config) => {
    const token = await getCurrentUserToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Types
export interface User {
  id: number;
  name: string;
  email: string;
  role: string;
  google_id: string;
}

export interface ChatMessage {
  id: number;
  message: string;
  response: string;
  timestamp: string;
  sources: any[];
}

export interface ChatResponse {
  response: string;
  chat_id: number;
  sources: any[];
  timestamp: string;
}

export interface PdfDocument {
  id: number;
  filename: string;
  upload_timestamp: string;
  chunk_count: number;
}

export interface FeedbackStats {
  total_chats: number;
  positive_feedback: number;
  negative_feedback: number;
  neutral_feedback: number;
  no_feedback: number;
  average_score: number | null;
}

// API functions

/**
 * Verify Firebase ID token and get/create user
 */
export const verifyToken = async (idToken: string, role: string = 'student'): Promise<User> => {
  const response = await apiClient.post('/api/auth/verify', { id_token: idToken, role });
  return response.data;
};

/**
 * Send chat message
 */
export const sendChatMessage = async (
  message: string,
  usePdfIds?: number[],
  topK: number = 3
): Promise<ChatResponse> => {
  const response = await apiClient.post('/api/chat', {
    message,
    use_pdf_ids: usePdfIds,
    top_k: topK,
    max_new_tokens: 200
  });
  return response.data;
};

/**
 * Get chat history
 */
export const getChatHistory = async (limit: number = 50, offset: number = 0): Promise<ChatMessage[]> => {
  const response = await apiClient.get('/api/history', {
    params: { limit, offset }
  });
  return response.data;
};

/**
 * Clear chat history
 */
export const clearChatHistory = async (): Promise<void> => {
  await apiClient.delete('/api/history/clear');
};

/**
 * Upload PDF file
 */
export const uploadPdfFile = async (file: File): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post('/api/pdf/upload_file', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Get user's PDFs
 */
export const getUserPdfs = async (): Promise<PdfDocument[]> => {
  const response = await apiClient.get('/api/pdf/user_pdfs');
  return response.data;
};

/**
 * Delete PDF
 */
export const deletePdf = async (pdfId: number): Promise<void> => {
  await apiClient.delete(`/api/pdf/pdf/${pdfId}`);
};

/**
 * Submit feedback
 */
export const submitFeedback = async (
  chatId: number,
  satisfactionScore: number,
  comment?: string
): Promise<void> => {
  await apiClient.post('/api/feedback/submit', {
    chat_id: chatId,
    satisfaction_score: satisfactionScore,
    comment
  });
};

/**
 * Get feedback stats (professor)
 */
export const getFeedbackStats = async (days: number = 30): Promise<FeedbackStats> => {
  const response = await apiClient.get('/api/professor/stats', {
    params: { days }
  });
  return response.data;
};

/**
 * Get low-rated chats (professor)
 */
export const getLowRatedChats = async (days: number = 30, limit: number = 100): Promise<any[]> => {
  const response = await apiClient.get('/api/professor/low_rated', {
    params: { days, limit }
  });
  return response.data;
};

/**
 * Get Gemini summary (professor)
 */
export const getGeminiSummary = async (days: number = 30): Promise<any> => {
  const response = await apiClient.get('/api/professor/summary', {
    params: { days }
  });
  return response.data;
};

/**
 * Export low-rated chats as CSV
 */
export const exportLowRatedCsv = async (days: number = 30): Promise<Blob> => {
  const response = await apiClient.get('/api/professor/export_csv', {
    params: { days },
    responseType: 'blob'
  });
  return response.data;
};

export default apiClient;

