/**
 * ChatBox component for displaying messages and handling user input
 */
import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User as UserIcon } from 'lucide-react';
import { sendChatMessage, ChatMessage } from '../services/api';
import FeedbackWidget from './FeedbackWidget';

interface ChatBoxProps {
  selectedPdfIds?: number[];
}

const ChatBox: React.FC<ChatBoxProps> = ({ selectedPdfIds }) => {
  const [messages, setMessages] = useState<Array<ChatMessage & { role: string }>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

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
      // Send to API
      const response = await sendChatMessage(userMessage, selectedPdfIds);

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

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <Bot size={64} className="mb-4 opacity-50" />
            <p className="text-lg">Ask me anything about Data Structures & Algorithms!</p>
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-2 max-w-2xl">
              <button
                onClick={() => setInput("Explain binary search trees")}
                className="p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg text-left transition-colors"
              >
                Explain binary search trees
              </button>
              <button
                onClick={() => setInput("What is time complexity?")}
                className="p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg text-left transition-colors"
              >
                What is time complexity?
              </button>
              <button
                onClick={() => setInput("Compare quicksort and mergesort")}
                className="p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg text-left transition-colors"
              >
                Compare quicksort and mergesort
              </button>
              <button
                onClick={() => setInput("Explain dynamic programming")}
                className="p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg text-left transition-colors"
              >
                Explain dynamic programming
              </button>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  msg.role === 'user' ? 'bg-indigo-500' : 'bg-green-500'
                }`}>
                  {msg.role === 'user' ? (
                    <UserIcon size={20} className="text-white" />
                  ) : (
                    <Bot size={20} className="text-white" />
                  )}
                </div>

                {/* Message Content */}
                <div className="flex-1">
                  <div className={`p-4 rounded-lg ${
                    msg.role === 'user' 
                      ? 'bg-indigo-500 text-white' 
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    <p className="whitespace-pre-wrap">{msg.role === 'user' ? msg.message : msg.response}</p>
                  </div>

                  {/* Sources */}
                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 text-xs text-gray-500">
                      <p className="font-medium mb-1">Sources:</p>
                      {msg.sources.map((source: any, sidx: number) => (
                        <p key={sidx}>📄 {source.filename} (chunk {source.chunk_index})</p>
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

      {/* Input Area */}
      <div className="border-t p-4">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask a DSA question..."
            className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
            rows={2}
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-6 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatBox;

