/**
 * Student page with chat interface, PDF upload, and history
 * Requires students to join a class before chatting
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, BookOpen, Plus, GraduationCap } from 'lucide-react';
import { logOut } from '../services/firebase';
import ChatBox from '../components/ChatBox';
import HistoryList from '../components/HistoryList';
import JoinClassModal from '../components/JoinClassModal';
import { getEnrolledClasses, EnrolledClass } from '../services/api';

const Student: React.FC = () => {
  const navigate = useNavigate();
  const [refreshKey, setRefreshKey] = useState(0);
  const [enrolledClasses, setEnrolledClasses] = useState<EnrolledClass[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [showJoinModal, setShowJoinModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const userData = JSON.parse(localStorage.getItem('user') || '{}');

  useEffect(() => {
    loadEnrolledClasses();
  }, []);

  const loadEnrolledClasses = async () => {
    try {
      setLoading(true);
      const classes = await getEnrolledClasses();
      setEnrolledClasses(classes);
      
      // Auto-select first class if available and none selected
      if (classes.length > 0 && !selectedClassId) {
        setSelectedClassId(classes[0].id);
      }
    } catch (error) {
      console.error('Error loading enrolled classes:', error);
    } finally {
      setLoading(false);
    }
  };

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

  const handleJoinSuccess = () => {
    setShowJoinModal(false);
    loadEnrolledClasses();
  };

  const selectedClass = enrolledClasses.find(c => c.id === selectedClassId);

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
                <p className="text-sm text-gray-500">Your Learning Assistant</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {/* Class Selector */}
              {enrolledClasses.length > 0 && (
                <div className="flex items-center gap-2">
                  <select
                    value={selectedClassId || ''}
                    onChange={(e) => setSelectedClassId(Number(e.target.value))}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">Select a class</option>
                    {enrolledClasses.map((cls) => (
                      <option key={cls.id} value={cls.id}>
                        {cls.name}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={() => setShowJoinModal(true)}
                    className="flex items-center gap-2 px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                    title="Join another class"
                  >
                    <Plus size={16} />
                    Join Class
                  </button>
                </div>
              )}
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
        {loading ? (
          <div className="flex items-center justify-center h-[calc(100vh-200px)]">
            <div className="text-gray-600">Loading classes...</div>
          </div>
        ) : enrolledClasses.length === 0 ? (
          /* No classes enrolled - show prompt to join */
          <div className="flex items-center justify-center h-[calc(100vh-200px)]">
            <div className="bg-white rounded-lg shadow-sm p-12 text-center max-w-md">
              <GraduationCap className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">No Classes Joined</h3>
              <p className="text-gray-600 mb-6">
                You need to join a class before you can start chatting with the AI. 
                Ask your professor for a class code to get started.
              </p>
              <button
                onClick={() => setShowJoinModal(true)}
                className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors inline-flex items-center gap-2"
              >
                <Plus className="w-5 h-5" />
                Join Your First Class
              </button>
            </div>
          </div>
        ) : !selectedClassId ? (
          /* Classes exist but none selected */
          <div className="flex items-center justify-center h-[calc(100vh-200px)]">
            <div className="bg-white rounded-lg shadow-sm p-12 text-center max-w-md">
              <BookOpen className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Select a Class</h3>
              <p className="text-gray-600">
                Choose a class from the dropdown above to start chatting with the AI.
              </p>
            </div>
          </div>
        ) : (
          /* Class selected - show chat interface */
          <div className="space-y-4">
            {/* Selected Class Info */}
            <div className="bg-white rounded-lg shadow-sm p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">{selectedClass?.name}</h2>
                  {selectedClass?.description && (
                    <p className="text-sm text-gray-600 mt-1">{selectedClass.description}</p>
                  )}
                </div>
                <div className="text-sm text-gray-500">
                  Professor: {selectedClass?.professor_name}
                </div>
              </div>
            </div>

            {/* Chat and Sidebar Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 h-[calc(100vh-300px)]">
              {/* Left Sidebar - Chat History & PDFs */}
              <div className="lg:col-span-1 h-full overflow-hidden">
                <HistoryList key={refreshKey} classId={selectedClassId} />
              </div>

              {/* Main Chat Area - Wider */}
              <div className="lg:col-span-4 h-full">
                <ChatBox classId={selectedClassId} onUploadSuccess={handleUploadSuccess} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Join Class Modal */}
      <JoinClassModal
        isOpen={showJoinModal}
        onClose={() => setShowJoinModal(false)}
        onSuccess={handleJoinSuccess}
      />
    </div>
  );
};

export default Student;
