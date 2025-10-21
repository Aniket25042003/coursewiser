/**
 * History list component for displaying past conversations
 */
import React, { useState, useEffect } from 'react';
import { MessageSquare, Trash2, RefreshCw } from 'lucide-react';
import { getChatHistory, clearChatHistory, ChatMessage, getUserPdfs, PdfDocument, deletePdf } from '../services/api';

interface HistoryListProps {
  onRefresh?: () => void;
  classId?: number | null;
}

const HistoryList: React.FC<HistoryListProps> = ({ onRefresh, classId }) => {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [pdfs, setPdfs] = useState<PdfDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'chats' | 'pdfs'>('chats');

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await getChatHistory(20);
      setHistory(data);
    } catch (error) {
      console.error('Error loading history:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadPdfs = async () => {
    if (!classId) return;
    setLoading(true);
    try {
      const data = await getUserPdfs(classId);
      setPdfs(data);
    } catch (error) {
      console.error('Error loading PDFs:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'chats') {
      loadHistory();
    } else {
      loadPdfs();
    }
  }, [activeTab, classId]);

  const handleClearHistory = async () => {
    if (!confirm('Are you sure you want to clear all chat history?')) return;

    try {
      await clearChatHistory();
      setHistory([]);
      if (onRefresh) onRefresh();
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  };

  const handleDeletePdf = async (pdfId: number) => {
    if (!confirm('Are you sure you want to delete this PDF?')) return;

    try {
      await deletePdf(pdfId);
      setPdfs(pdfs.filter(p => p.id !== pdfId));
    } catch (error) {
      console.error('Error deleting PDF:', error);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 h-full flex flex-col">
      {/* Header with Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1 p-2 bg-gray-50">
          <button
            onClick={() => setActiveTab('chats')}
            className={`flex-1 px-3 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'chats'
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            💬 Chats
          </button>
          <button
            onClick={() => setActiveTab('pdfs')}
            className={`flex-1 px-3 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'pdfs'
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            📄 PDFs
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'chats' ? (
          <>
            {/* Chat History */}
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
              </div>
            ) : history.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <MessageSquare size={48} className="mx-auto mb-3 opacity-50" />
                <p>No chat history yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {history.map((chat) => (
                  <div
                    key={chat.id}
                    className="p-3 border border-gray-200 rounded-lg hover:border-indigo-300 hover:bg-indigo-50 transition-colors"
                  >
                    <p className="text-sm font-medium text-gray-800 mb-1 line-clamp-2">
                      {chat.message}
                    </p>
                    <p className="text-xs text-gray-500 line-clamp-2">
                      {chat.response}
                    </p>
                    <p className="text-xs text-gray-400 mt-2">
                      {new Date(chat.timestamp).toLocaleString()}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            {/* PDFs */}
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
              </div>
            ) : pdfs.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <MessageSquare size={48} className="mx-auto mb-3 opacity-50" />
                <p>No PDFs uploaded yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {pdfs.map((pdf) => (
                  <div
                    key={pdf.id}
                    className="p-3 border border-gray-200 rounded-lg hover:border-indigo-300 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-800 mb-1">
                          📄 {pdf.filename}
                        </p>
                        <p className="text-xs text-gray-500">
                          {pdf.chunk_count} chunks indexed
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          {new Date(pdf.upload_timestamp).toLocaleString()}
                        </p>
                      </div>
                      <button
                        onClick={() => handleDeletePdf(pdf.id)}
                        className="text-red-500 hover:text-red-700 p-1"
                        title="Delete PDF"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="border-t p-3 flex gap-2">
        <button
          onClick={activeTab === 'chats' ? loadHistory : loadPdfs}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded transition-colors"
        >
          <RefreshCw size={16} />
          Refresh
        </button>
        {activeTab === 'chats' && history.length > 0 && (
          <button
            onClick={handleClearHistory}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-red-50 text-red-600 hover:bg-red-100 rounded transition-colors"
          >
            <Trash2 size={16} />
            Clear All
          </button>
        )}
      </div>
    </div>
  );
};

export default HistoryList;

