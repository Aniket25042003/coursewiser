/**
 * Professor dashboard page with feedback analytics and Gemini summaries
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, BookOpen, TrendingDown, Download, Sparkles, BarChart3 } from 'lucide-react';
import { logOut } from '../services/firebase';
import { getFeedbackStats, getLowRatedChats, getGeminiSummary, exportLowRatedCsv } from '../services/api';

const Professor: React.FC = () => {
  const navigate = useNavigate();
  const userData = JSON.parse(localStorage.getItem('user') || '{}');
  const [stats, setStats] = useState<any>(null);
  const [lowRatedChats, setLowRatedChats] = useState<any[]>([]);
  const [geminiSummary, setGeminiSummary] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [days, setDays] = useState(30);

  useEffect(() => {
    loadStats();
    loadLowRatedChats();
  }, [days]);

  const loadStats = async () => {
    try {
      const data = await getFeedbackStats(days);
      setStats(data);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const loadLowRatedChats = async () => {
    try {
      const data = await getLowRatedChats(days, 50);
      setLowRatedChats(data);
    } catch (error) {
      console.error('Error loading low-rated chats:', error);
    }
  };

  const handleGenerateSummary = async () => {
    setLoading(true);
    try {
      const data = await getGeminiSummary(days);
      setGeminiSummary(data.summary);
    } catch (error: any) {
      console.error('Error generating summary:', error);
      alert(error.response?.data?.detail || 'Failed to generate summary');
    } finally {
      setLoading(false);
    }
  };

  const handleExportCsv = async () => {
    try {
      const blob = await exportLowRatedCsv(days);
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
      await logOut();
      localStorage.removeItem('user');
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
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
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
      </div>
    </div>
  );
};

export default Professor;

