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
    // Try professor token first (JWT)
    const professorToken = localStorage.getItem('professorToken');
    if (professorToken) {
      config.headers.Authorization = `Bearer ${professorToken}`;
    } else {
      // Fall back to Firebase token for students
      const token = await getCurrentUserToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
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
      // Unauthorized - clear auth and redirect to login
      // Only redirect if not already on login page to prevent infinite loop
      if (!window.location.pathname.includes('/login')) {
        localStorage.removeItem('user');
        localStorage.removeItem('professorToken');
        window.location.href = '/login';
      }
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
  google_id?: string;
  username?: string;
  must_change_password?: boolean;
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

export interface Class {
  id: number;
  name: string;
  description?: string;
  class_code: string;
  professor_id: number;
  professor_name: string;
  created_at: string;
  is_active: boolean;
  enrolled_count?: number;
}

export interface EnrolledClass {
  id: number;
  name: string;
  description?: string;
  professor_name: string;
  joined_at: string;
}

export interface ClassMaterial {
  id: number;
  filename: string;
  upload_timestamp: string;
  class_id: number;
}

export interface ProfessorLoginResponse {
  user: User;
  access_token: string;
  token_type: string;
}

// API functions

/**
 * Verify Firebase ID token and get/create student user
 */
export const verifyToken = async (idToken: string): Promise<User> => {
  const response = await apiClient.post('/api/auth/verify', { id_token: idToken });
  return response.data;
};

/**
 * Professor login with username and password
 */
export const loginProfessor = async (username: string, password: string): Promise<ProfessorLoginResponse> => {
  const response = await apiClient.post('/api/auth/professor/login', { username, password });
  // Store the JWT token in localStorage
  if (response.data.access_token) {
    localStorage.setItem('professorToken', response.data.access_token);
  }
  return response.data;
};

/**
 * Change professor password
 */
export const changePassword = async (oldPassword: string, newPassword: string): Promise<void> => {
  await apiClient.post('/api/auth/professor/change_password', {
    old_password: oldPassword,
    new_password: newPassword
  });
};

/**
 * Send chat message
 */
export const sendChatMessage = async (
  message: string,
  classId: number,
  usePdfIds?: number[],
  topK: number = 3
): Promise<ChatResponse> => {
  const response = await apiClient.post('/api/chat', {
    message,
    class_id: classId,
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
export const uploadPdfFile = async (file: File, classId: number): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('class_id', classId.toString());
  
  const response = await apiClient.post('/api/pdf/upload_file', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Get user's PDFs for a specific class
 */
export const getUserPdfs = async (classId?: number): Promise<PdfDocument[]> => {
  const params = classId ? { class_id: classId } : {};
  const response = await apiClient.get('/api/pdf/user_pdfs', { params });
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
export const getFeedbackStats = async (classId: number, days: number = 30): Promise<FeedbackStats> => {
  const response = await apiClient.get('/api/professor/stats', {
    params: { class_id: classId, days }
  });
  return response.data;
};

/**
 * Get low-rated chats (professor)
 */
export const getLowRatedChats = async (classId: number, days: number = 30, limit: number = 100): Promise<any[]> => {
  const response = await apiClient.get('/api/professor/low_rated', {
    params: { class_id: classId, days, limit }
  });
  return response.data;
};

/**
 * Get Gemini summary (professor)
 */
export const getGeminiSummary = async (classId: number, days: number = 30): Promise<any> => {
  const response = await apiClient.get('/api/professor/summary', {
    params: { class_id: classId, days }
  });
  return response.data;
};

/**
 * Export low-rated chats as CSV
 */
export const exportLowRatedCsv = async (classId: number, days: number = 30): Promise<Blob> => {
  const response = await apiClient.get('/api/professor/export_csv', {
    params: { class_id: classId, days },
    responseType: 'blob'
  });
  return response.data;
};

// Class Management API functions

/**
 * Create a new class (professor)
 */
export const createClass = async (name: string, description?: string): Promise<Class> => {
  const response = await apiClient.post('/api/classes', { name, description });
  return response.data;
};

/**
 * Get professor's classes
 */
export const getMyClasses = async (): Promise<Class[]> => {
  const response = await apiClient.get('/api/classes/my_classes');
  return response.data;
};

/**
 * Update class details (professor)
 */
export const updateClass = async (
  classId: number,
  name?: string,
  description?: string,
  isActive?: boolean
): Promise<Class> => {
  const response = await apiClient.put(`/api/classes/${classId}`, {
    name,
    description,
    is_active: isActive
  });
  return response.data;
};

/**
 * Delete/deactivate class (professor)
 */
export const deleteClass = async (classId: number): Promise<void> => {
  await apiClient.delete(`/api/classes/${classId}`);
};

/**
 * Get students enrolled in a class (professor)
 */
export const getClassStudents = async (classId: number): Promise<any[]> => {
  const response = await apiClient.get(`/api/classes/${classId}/students`);
  return response.data;
};

/**
 * Join a class with class code (student)
 */
export const joinClass = async (classCode: string): Promise<EnrolledClass> => {
  const response = await apiClient.post('/api/classes/join', { class_code: classCode });
  return response.data;
};

/**
 * Get enrolled classes (student)
 */
export const getEnrolledClasses = async (): Promise<EnrolledClass[]> => {
  const response = await apiClient.get('/api/classes/enrolled');
  return response.data;
};

/**
 * Leave a class (student)
 */
export const leaveClass = async (classId: number): Promise<void> => {
  await apiClient.post(`/api/classes/leave/${classId}`);
};

/**
 * Upload class material (professor)
 */
export const uploadClassMaterial = async (classId: number, file: File): Promise<ClassMaterial> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(`/api/classes/${classId}/materials`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Get class materials
 */
export const getClassMaterials = async (classId: number): Promise<ClassMaterial[]> => {
  const response = await apiClient.get(`/api/classes/${classId}/materials`);
  return response.data;
};

/**
 * Delete class material (professor)
 */
export const deleteClassMaterial = async (classId: number, materialId: number): Promise<void> => {
  await apiClient.delete(`/api/classes/${classId}/materials/${materialId}`);
};

export default apiClient;

