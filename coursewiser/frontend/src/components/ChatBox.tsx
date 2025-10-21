/**
 * ChatBox component for displaying messages and handling user input
 */
import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User as UserIcon, Paperclip } from 'lucide-react';
import { sendChatMessage, ChatMessage, uploadPdfFile } from '../services/api';
import FeedbackWidget from './FeedbackWidget';

interface ChatBoxProps {
  selectedPdfIds?: number[];
  classId?: number | null;
  onUploadSuccess?: () => void;
}

const ChatBox: React.FC<ChatBoxProps> = ({ selectedPdfIds, classId, onUploadSuccess }) => {
  const [messages, setMessages] = useState<Array<ChatMessage & { role: string }>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');

    // Add user message to UI
    const newUserMessage = {
      id: Date.now(),
      role: 'user',
      message: userMessage,
      response: '',
      timestamp: new Date().toISOString(),
      sources: []
    };
    setMessages(prev => [...prev, newUserMessage]);

    setLoading(true);

    try {
      // Send to API - use classId if provided, otherwise use 1 as default
      if (!classId) {
        alert('Please select a class to start chatting');
        setLoading(false);
        return;
      }
      const response = await sendChatMessage(userMessage, classId, selectedPdfIds);

      // Add assistant response
      const assistantMessage = {
        id: response.chat_id,
        role: 'assistant',
        message: userMessage,
        response: response.response,
        timestamp: response.timestamp,
        sources: response.sources
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      console.error('Error sending message:', error);
      // Add error message
      const errorMessage = {
        id: Date.now(),
        role: 'assistant',
        message: userMessage,
        response: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
        sources: []
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Only PDF files are allowed');
      return;
    }

    if (!classId) {
      alert('Please select a class first');
      return;
    }

    setUploading(true);
    try {
      await uploadPdfFile(file, classId);
      if (onUploadSuccess) {
        onUploadSuccess();
      }
      alert('PDF uploaded successfully!');
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload PDF');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-xl shadow-lg border border-gray-200">
      {/* Messages Area - Scrollable */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4 shadow-lg">
              <Bot size={32} className="text-white" />
            </div>
            <p className="text-xl font-semibold text-gray-700 mb-2">Start a conversation</p>
            <p className="text-sm text-gray-500">Ask questions about your course materials</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-slide-up`}>
              <div className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-md ${
                  msg.role === 'user' 
                    ? 'bg-gradient-to-br from-indigo-500 to-purple-500' 
                    : 'bg-gradient-to-br from-green-500 to-emerald-500'
                }`}>
                  {msg.role === 'user' ? (
                    <UserIcon size={20} className="text-white" />
                  ) : (
                    <Bot size={20} className="text-white" />
                  )}
                </div>

                {/* Message Content */}
                <div className="flex-1">
                  <div className={`p-4 rounded-2xl shadow-sm ${
                    msg.role === 'user' 
                      ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white' 
                      : 'bg-gray-50 text-gray-800 border border-gray-200'
                  }`}>
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.role === 'user' ? msg.message : msg.response}</p>
                  </div>

                  {/* Sources */}
                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {msg.sources.map((source: any, sidx: number) => (
                        <div key={sidx} className="inline-flex items-center gap-1 px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-xs font-medium border border-indigo-200">
                          <span>📄</span>
                          <span>{source.filename}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Feedback Widget for Assistant Messages */}
                  {msg.role === 'assistant' && msg.id && (
                    <div className="mt-2">
                      <FeedbackWidget chatId={msg.id} />
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[80%]">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                <Bot size={20} className="text-white" />
              </div>
              <div className="flex-1 p-4 rounded-lg bg-gray-100">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Fixed at bottom with attachment button */}
      <div className="border-t border-gray-200 p-4 bg-gray-50">
        <div className="flex gap-2 items-end">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            className="hidden"
          />
          
          {/* Attachment button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="p-3 text-gray-600 hover:text-indigo-600 hover:bg-white rounded-xl transition-all disabled:opacity-50"
            title="Upload PDF"
          >
            {uploading ? (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-500"></div>
            ) : (
              <Paperclip size={22} />
            )}
          </button>
          
          {/* Text input */}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask your question..."
            className="flex-1 p-4 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white shadow-sm"
            rows={2}
            disabled={loading}
          />
          
          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="p-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-xl hover:from-indigo-600 hover:to-purple-600 shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center"
          >
            {loading ? (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatBox;

