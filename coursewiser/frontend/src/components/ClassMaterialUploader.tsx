/**
 * Class material uploader component for professors
 * Similar to PdfUploader but for class materials
 */
import React, { useState, useRef } from 'react';
import { Upload, File, Trash2, CheckCircle } from 'lucide-react';
import { uploadClassMaterial, getClassMaterials, deleteClassMaterial, type ClassMaterial } from '../services/api';

interface ClassMaterialUploaderProps {
  classId: number;
  onUploadSuccess?: () => void;
}

const ClassMaterialUploader: React.FC<ClassMaterialUploaderProps> = ({ classId, onUploadSuccess }) => {
  const [materials, setMaterials] = useState<ClassMaterial[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (classId) {
      loadMaterials();
    }
  }, [classId]);

  const loadMaterials = async () => {
    try {
      setLoading(true);
      const data = await getClassMaterials(classId);
      setMaterials(data);
    } catch (err: any) {
      console.error('Error loading materials:', err);
      setError('Failed to load materials');
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.name.endsWith('.pdf')) {
      setError('Only PDF files are allowed');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      await uploadClassMaterial(classId, file);
      setSuccess(`Successfully uploaded ${file.name}`);
      await loadMaterials();
      if (onUploadSuccess) {
        onUploadSuccess();
      }
      
      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      console.error('Error uploading material:', err);
      setError(err.response?.data?.detail || 'Failed to upload material');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (materialId: number, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    try {
      await deleteClassMaterial(classId, materialId);
      await loadMaterials();
      setSuccess(`Deleted ${filename}`);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      console.error('Error deleting material:', err);
      setError(err.response?.data?.detail || 'Failed to delete material');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Class Materials</h3>

      {/* Upload Button */}
      <div className="mb-4">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
          id="material-upload"
        />
        <label
          htmlFor="material-upload"
          className={`flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-indigo-300 rounded-lg cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-all ${
            uploading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {uploading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-600"></div>
              <span className="text-indigo-600">Uploading...</span>
            </>
          ) : (
            <>
              <Upload size={20} className="text-indigo-600" />
              <span className="text-indigo-600 font-medium">Upload PDF Material</span>
            </>
          )}
        </label>
      </div>

      {/* Success Message */}
      {success && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
          <CheckCircle size={16} className="text-green-600" />
          <span className="text-green-700 text-sm">{success}</span>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Materials List */}
      <div className="space-y-2">
        {loading ? (
          <div className="text-center py-4 text-gray-500">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600 mx-auto mb-2"></div>
            <p className="text-sm">Loading materials...</p>
          </div>
        ) : materials.length === 0 ? (
          <p className="text-gray-500 text-sm italic text-center py-4">
            No materials uploaded yet
          </p>
        ) : (
          materials.map((material) => (
            <div
              key={material.id}
              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center gap-3">
                <File size={20} className="text-gray-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">{material.filename}</p>
                  <p className="text-xs text-gray-500">
                    {new Date(material.upload_timestamp).toLocaleDateString()}
                  </p>
                </div>
              </div>
              <button
                onClick={() => handleDelete(material.id, material.filename)}
                className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                title="Delete material"
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))
        )}
      </div>

      <p className="text-xs text-gray-500 mt-4">
        Uploaded materials will be available to all students enrolled in this class.
      </p>
    </div>
  );
};

export default ClassMaterialUploader;

