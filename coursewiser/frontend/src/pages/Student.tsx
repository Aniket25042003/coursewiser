/**
 * Student page with chat interface, PDF upload, and history
 */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, BookOpen } from 'lucide-react';
import { logOut } from '../services/firebase';
import ChatBox from '../components/ChatBox';
import PdfUploader from '../components/PdfUploader';
import HistoryList from '../components/HistoryList';

const Student: React.FC = () => {
  const navigate = useNavigate();
  const [refreshKey, setRefreshKey] = useState(0);
  const userData = JSON.parse(localStorage.getItem('user') || '{}');

  const handleLogout = async () => {
    try {
      await logOut();
      localStorage.removeItem('user');
      navigate('/login');
    } catch (error) {
      console.error('Error logging out:', error);
    }
  };

  const handleUploadSuccess = () => {
    // Refresh the history list
    setRefreshKey(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-3">
              <BookOpen size={32} className="text-indigo-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">CourseWiser</h1>
                <p className="text-sm text-gray-500">DSA Learning Assistant</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">{userData.name}</p>
                <p className="text-xs text-gray-500">{userData.email}</p>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                <LogOut size={16} />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-140px)]">
          {/* Left Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            <PdfUploader onUploadSuccess={handleUploadSuccess} />
            <div className="h-[calc(100%-250px)]">
              <HistoryList key={refreshKey} />
            </div>
          </div>

          {/* Main Chat Area */}
          <div className="lg:col-span-3">
            <ChatBox />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Student;

