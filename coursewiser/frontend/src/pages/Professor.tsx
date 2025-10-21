/**
 * Professor dashboard page with feedback analytics and Gemini summaries
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, BookOpen, TrendingDown, Download, Sparkles, BarChart3, GraduationCap } from 'lucide-react';
import { getFeedbackStats, getLowRatedChats, getGeminiSummary, exportLowRatedCsv, getMyClasses, Class } from '../services/api';

const Professor: React.FC = () => {
  const navigate = useNavigate();
  const userData = JSON.parse(localStorage.getItem('user') || '{}');
  const [stats, setStats] = useState<any>(null);
  const [lowRatedChats, setLowRatedChats] = useState<any[]>([]);
  const [geminiSummary, setGeminiSummary] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [days, setDays] = useState(30);
  const [classes, setClasses] = useState<Class[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);

  useEffect(() => {
    loadClasses();
  }, []);

  useEffect(() => {
    if (selectedClassId) {
      loadStats();
      loadLowRatedChats();
    }
  }, [days, selectedClassId]);

  const loadClasses = async () => {
    try {
      const data = await getMyClasses();
      setClasses(data);
      if (data.length > 0 && !selectedClassId) {
        setSelectedClassId(data[0].id);
      }
    } catch (error) {
      console.error('Error loading classes:', error);
    }
  };

  const loadStats = async () => {
    if (!selectedClassId) return;
    try {
      const data = await getFeedbackStats(selectedClassId, days);
      setStats(data);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const loadLowRatedChats = async () => {
    if (!selectedClassId) return;
    try {
      const data = await getLowRatedChats(selectedClassId, days, 50);
      setLowRatedChats(data);
    } catch (error) {
      console.error('Error loading low-rated chats:', error);
    }
  };

  const handleGenerateSummary = async () => {
    if (!selectedClassId) return;
    setLoading(true);
    try {
      const data = await getGeminiSummary(selectedClassId, days);
      setGeminiSummary(data.summary);
    } catch (error: any) {
      console.error('Error generating summary:', error);
      alert(error.response?.data?.detail || 'Failed to generate summary');
    } finally {
      setLoading(false);
    }
  };

  const handleExportCsv = async () => {
    if (!selectedClassId) return;
    try {
      const blob = await exportLowRatedCsv(selectedClassId, days);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `low_rated_feedback_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting CSV:', error);
    }
  };

  const handleLogout = async () => {
    try {
      localStorage.removeItem('user');
      localStorage.removeItem('professorToken');
      window.dispatchEvent(new Event('logout'));
      navigate('/login');
    } catch (error) {
      console.error('Error logging out:', error);
    }
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
                <h1 className="text-2xl font-bold text-gray-900">Professor Dashboard</h1>
                <p className="text-sm text-gray-500">CourseWiser Analytics</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {/* Class Selector */}
              {classes.length > 0 && (
                <select
                  value={selectedClassId || ''}
                  onChange={(e) => setSelectedClassId(Number(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="">Select a class</option>
                  {classes.map((cls) => (
                    <option key={cls.id} value={cls.id}>
                      {cls.name}
                    </option>
                  ))}
                </select>
              )}
              <select
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value={7}>Last 7 days</option>
                <option value={30}>Last 30 days</option>
                <option value={90}>Last 90 days</option>
              </select>
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">{userData.name}</p>
                <p className="text-xs text-gray-500">Professor</p>
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
          
          {/* Navigation Tabs */}
          <div className="flex gap-4 border-t pt-2">
            <button
              onClick={() => navigate('/professor')}
              className="px-4 py-2 text-sm font-medium text-indigo-600 border-b-2 border-indigo-600"
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              Analytics Dashboard
            </button>
            <button
              onClick={() => navigate('/professor/classes')}
              className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-indigo-600 hover:border-b-2 hover:border-indigo-600"
            >
              <GraduationCap className="w-4 h-4 inline mr-2" />
              Class Management
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* No class selected message */}
        {classes.length === 0 ? (
          <div className="bg-white rounded-lg shadow-sm p-12 text-center">
            <GraduationCap className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No Classes Yet</h3>
            <p className="text-gray-600 mb-6">Create your first class to start tracking analytics</p>
            <button
              onClick={() => navigate('/professor/classes')}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors inline-flex items-center gap-2"
            >
              <GraduationCap className="w-5 h-5" />
              Go to Class Management
            </button>
          </div>
        ) : !selectedClassId ? (
          <div className="bg-white rounded-lg shadow-sm p-12 text-center">
            <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Select a Class</h3>
            <p className="text-gray-600">Choose a class from the dropdown above to view analytics</p>
          </div>
        ) : (
          <>
            {/* Stats Cards */}
            {stats && (
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total Chats</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_chats}</p>
                </div>
                <BarChart3 className="text-gray-400" size={32} />
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Positive</p>
                  <p className="text-2xl font-bold text-green-600">{stats.positive_feedback}</p>
                </div>
                <div className="text-green-500 text-2xl">👍</div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Negative</p>
                  <p className="text-2xl font-bold text-red-600">{stats.negative_feedback}</p>
                </div>
                <div className="text-red-500 text-2xl">👎</div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">No Feedback</p>
                  <p className="text-2xl font-bold text-gray-600">{stats.no_feedback}</p>
                </div>
                <div className="text-gray-400 text-2xl">⚪</div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Avg Score</p>
                  <p className="text-2xl font-bold text-indigo-600">
                    {stats.average_score?.toFixed(2) || 'N/A'}
                  </p>
                </div>
                <div className="text-indigo-500 text-2xl">📊</div>
              </div>
            </div>
              </div>
            )}

            {/* AI Summary Section */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                  <Sparkles className="text-purple-500" size={24} />
                  AI-Powered Insights
                </h2>
                <button
                  onClick={handleGenerateSummary}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      Generate Summary
                    </>
                  )}
                </button>
              </div>

              {geminiSummary ? (
                <div className="prose max-w-none">
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <pre className="whitespace-pre-wrap text-sm text-gray-800 font-sans">
                      {geminiSummary}
                    </pre>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500 italic">
                  Click "Generate Summary" to get AI-powered insights from Gemini about common student issues.
                </p>
              )}
            </div>

            {/* Low-Rated Chats */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                  <TrendingDown className="text-red-500" size={24} />
                  Low-Rated Chats ({lowRatedChats.length})
                </h2>
                <button
                  onClick={handleExportCsv}
                  className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors"
                >
                  <Download size={16} />
                  Export CSV
                </button>
              </div>

              {lowRatedChats.length === 0 ? (
                <p className="text-gray-500 italic">No low-rated feedback in the selected period.</p>
              ) : (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {lowRatedChats.slice(0, 20).map((chat) => (
                    <div key={chat.chat_id} className="border border-gray-200 rounded-lg p-4 hover:border-red-300 transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <p className="text-sm text-gray-500">
                            Student: {chat.student_name} • {new Date(chat.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <span className="text-red-500">👎</span>
                      </div>
                      <div className="space-y-2">
                        <div>
                          <p className="text-xs font-semibold text-gray-700">Question:</p>
                          <p className="text-sm text-gray-800">{chat.message}</p>
                        </div>
                        <div>
                          <p className="text-xs font-semibold text-gray-700">Response:</p>
                          <p className="text-sm text-gray-600 line-clamp-3">{chat.response}</p>
                        </div>
                        {chat.comment && (
                          <div>
                            <p className="text-xs font-semibold text-gray-700">Student Comment:</p>
                            <p className="text-sm text-red-600 italic">{chat.comment}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Professor;

