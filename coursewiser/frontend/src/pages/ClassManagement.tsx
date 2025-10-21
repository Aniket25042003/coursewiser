import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Copy, Trash2, Upload, X, Users, BookOpen, Check, LogOut, BarChart3, GraduationCap } from 'lucide-react';
import { 
  createClass, 
  getMyClasses, 
  deleteClass, 
  uploadClassMaterial, 
  getClassMaterials, 
  deleteClassMaterial,
  getClassStudents,
  Class,
  ClassMaterial
} from '../services/api';

interface Student {
  id: number;
  name: string;
  email: string;
  joined_at: string;
}

const ClassManagement: React.FC = () => {
  const navigate = useNavigate();
  const userData = JSON.parse(localStorage.getItem('user') || '{}');
  const [classes, setClasses] = useState<Class[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedClass, setSelectedClass] = useState<Class | null>(null);
  const [showMaterialsModal, setShowMaterialsModal] = useState(false);
  const [showStudentsModal, setShowStudentsModal] = useState(false);
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  // Create class form state
  const [newClassName, setNewClassName] = useState('');
  const [newClassDescription, setNewClassDescription] = useState('');
  const [createError, setCreateError] = useState<string | null>(null);

  // Materials state
  const [materials, setMaterials] = useState<ClassMaterial[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Students state
  const [students, setStudents] = useState<Student[]>([]);
  const [loadingStudents, setLoadingStudents] = useState(false);

  useEffect(() => {
    loadClasses();
  }, []);

  const loadClasses = async () => {
    try {
      setLoading(true);
      const data = await getMyClasses();
      setClasses(data);
    } catch (error) {
      console.error('Failed to load classes:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateClass = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreateError(null);

    if (!newClassName.trim()) {
      setCreateError('Class name is required');
      return;
    }

    try {
      await createClass(newClassName, newClassDescription);
      
      setNewClassName('');
      setNewClassDescription('');
      setShowCreateModal(false);
      loadClasses();
    } catch (error: any) {
      setCreateError(error.response?.data?.detail || 'Failed to create class');
    }
  };

  const handleDeleteClass = async (classId: number) => {
    if (!confirm('Are you sure you want to delete this class? This action cannot be undone.')) {
      return;
    }

    try {
      await deleteClass(classId);
      loadClasses();
    } catch (error) {
      console.error('Failed to delete class:', error);
      alert('Failed to delete class');
    }
  };

  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const handleViewMaterials = async (cls: Class) => {
    setSelectedClass(cls);
    setShowMaterialsModal(true);
    try {
      const data = await getClassMaterials(cls.id);
      setMaterials(data);
    } catch (error) {
      console.error('Failed to load materials:', error);
    }
  };

  const handleViewStudents = async (cls: Class) => {
    setSelectedClass(cls);
    setShowStudentsModal(true);
    setLoadingStudents(true);
    try {
      const data = await getClassStudents(cls.id);
      setStudents(data);
    } catch (error) {
      console.error('Failed to load students:', error);
    } finally {
      setLoadingStudents(false);
    }
  };

  const handleUploadMaterial = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !selectedClass) return;

    const file = e.target.files[0];
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Only PDF files are allowed');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      await uploadClassMaterial(selectedClass.id, file);
      const data = await getClassMaterials(selectedClass.id);
      setMaterials(data);
    } catch (error: any) {
      setUploadError(error.response?.data?.detail || 'Failed to upload material');
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteMaterial = async (materialId: number) => {
    if (!selectedClass || !confirm('Delete this material?')) return;

    try {
      await deleteClassMaterial(selectedClass.id, materialId);
      const data = await getClassMaterials(selectedClass.id);
      setMaterials(data);
    } catch (error) {
      console.error('Failed to delete material:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    localStorage.removeItem('professorToken');
    window.dispatchEvent(new Event('logout'));
    navigate('/login');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-gray-600">Loading classes...</div>
      </div>
    );
  }

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
                <p className="text-sm text-gray-500">Class Management</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
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
              className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-indigo-600 hover:border-b-2 hover:border-indigo-600"
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              Analytics Dashboard
            </button>
            <button
              onClick={() => navigate('/professor/classes')}
              className="px-4 py-2 text-sm font-medium text-indigo-600 border-b-2 border-indigo-600"
            >
              <GraduationCap className="w-4 h-4 inline mr-2" />
              Class Management
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Your Classes</h2>
            <p className="text-gray-600 mt-2">Create and manage your classes</p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            Create New Class
          </button>
        </div>

        {/* Classes Grid */}
        {classes.length === 0 ? (
          <div className="bg-white rounded-lg shadow-sm p-12 text-center">
            <BookOpen className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No classes yet</h3>
            <p className="text-gray-600 mb-6">Create your first class to get started</p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors inline-flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              Create New Class
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {classes.map((cls) => (
              <div key={cls.id} className="bg-white rounded-xl shadow-md p-6 hover:shadow-xl transform hover:scale-[1.02] transition-all duration-300 border border-gray-100 group">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-xl font-semibold text-gray-900">{cls.name}</h3>
                  <button
                    onClick={() => handleDeleteClass(cls.id)}
                    className="text-red-600 hover:text-red-700 p-1"
                    title="Delete class"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>

                {cls.description && (
                  <p className="text-gray-600 text-sm mb-4 line-clamp-2">{cls.description}</p>
                )}

                {/* Class Code */}
                <div className="mb-4 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border-2 border-indigo-100 group-hover:border-indigo-200 transition-colors">
                  <div className="text-xs text-indigo-600 font-semibold mb-2 uppercase tracking-wide">Class Code</div>
                  <div className="flex items-center justify-between">
                    <code className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent tracking-wider">{cls.class_code}</code>
                    <button
                      onClick={() => handleCopyCode(cls.class_code)}
                      className="text-indigo-600 hover:text-indigo-700 p-2 hover:bg-white/50 rounded-lg transition-all"
                      title="Copy class code"
                    >
                      {copiedCode === cls.class_code ? (
                        <Check className="w-5 h-5 text-green-600 animate-bounce" />
                      ) : (
                        <Copy className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Stats */}
                <div className="flex gap-4 mb-4 text-sm text-gray-600">
                  <div>
                    <Users className="w-4 h-4 inline mr-1" />
                    {cls.enrolled_count} student{cls.enrolled_count !== 1 ? 's' : ''}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  <button
                    onClick={() => handleViewMaterials(cls)}
                    className="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-lg font-medium hover:bg-gray-200 transition-colors text-sm flex items-center justify-center gap-2"
                  >
                    <Upload className="w-4 h-4" />
                    Materials
                  </button>
                  <button
                    onClick={() => handleViewStudents(cls)}
                    className="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-lg font-medium hover:bg-gray-200 transition-colors text-sm flex items-center justify-center gap-2"
                  >
                    <Users className="w-4 h-4" />
                    Students
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Class Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-gray-900">Create New Class</h2>
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setCreateError(null);
                  setNewClassName('');
                  setNewClassDescription('');
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <form onSubmit={handleCreateClass}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Class Name *
                </label>
                <input
                  type="text"
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="e.g., Data Structures 101"
                  required
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description (optional)
                </label>
                <textarea
                  value={newClassDescription}
                  onChange={(e) => setNewClassDescription(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Brief description of the class"
                  rows={3}
                />
              </div>

              {createError && (
                <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm">
                  {createError}
                </div>
              )}

              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowCreateModal(false);
                    setCreateError(null);
                    setNewClassName('');
                    setNewClassDescription('');
                  }}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 bg-indigo-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-indigo-700 transition-colors"
                >
                  Create Class
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Materials Modal */}
      {showMaterialsModal && selectedClass && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full p-6 max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-gray-900">
                Materials: {selectedClass.name}
              </h2>
              <button
                onClick={() => {
                  setShowMaterialsModal(false);
                  setSelectedClass(null);
                  setUploadError(null);
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Upload Section */}
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload New Material (PDF only)
              </label>
              <input
                type="file"
                accept=".pdf"
                onChange={handleUploadMaterial}
                disabled={uploading}
                className="w-full"
              />
              {uploading && (
                <p className="text-sm text-indigo-600 mt-2">Uploading and processing...</p>
              )}
              {uploadError && (
                <p className="text-sm text-red-600 mt-2">{uploadError}</p>
              )}
            </div>

            {/* Materials List */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                Uploaded Materials ({materials.length})
              </h3>
              {materials.length === 0 ? (
                <p className="text-gray-600 text-center py-8">No materials uploaded yet</p>
              ) : (
                <div className="space-y-2">
                  {materials.map((material) => (
                    <div
                      key={material.id}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100"
                    >
                      <div className="flex-1">
                        <p className="font-medium text-gray-900">{material.filename}</p>
                        <p className="text-sm text-gray-500">
                          Uploaded {new Date(material.upload_timestamp).toLocaleDateString()}
                        </p>
                      </div>
                      <button
                        onClick={() => handleDeleteMaterial(material.id)}
                        className="text-red-600 hover:text-red-700 p-2"
                        title="Delete material"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Students Modal */}
      {showStudentsModal && selectedClass && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full p-6 max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-gray-900">
                Students: {selectedClass.name}
              </h2>
              <button
                onClick={() => {
                  setShowStudentsModal(false);
                  setSelectedClass(null);
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {loadingStudents ? (
              <p className="text-center text-gray-600 py-8">Loading students...</p>
            ) : students.length === 0 ? (
              <p className="text-center text-gray-600 py-8">No students enrolled yet</p>
            ) : (
              <div className="space-y-2">
                {students.map((student) => (
                  <div
                    key={student.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div>
                      <p className="font-medium text-gray-900">{student.name}</p>
                      <p className="text-sm text-gray-500">{student.email}</p>
                    </div>
                    <p className="text-sm text-gray-500">
                      Joined {new Date(student.joined_at).toLocaleDateString()}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ClassManagement;
