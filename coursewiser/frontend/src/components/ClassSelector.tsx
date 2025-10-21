/**
 * Class selector dropdown component
 * Shows enrolled classes for students, created classes for professors
 */
import React, { useEffect, useState } from 'react';
import { getEnrolledClasses, getMyClasses, type EnrolledClass, type Class } from '../services/api';

interface ClassSelectorProps {
  userRole: 'student' | 'professor';
  selectedClassId: number | null;
  onClassSelect: (classId: number) => void;
}

const ClassSelector: React.FC<ClassSelectorProps> = ({ userRole, selectedClassId, onClassSelect }) => {
  const [classes, setClasses] = useState<(EnrolledClass | Class)[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadClasses();
  }, [userRole]);

  const loadClasses = async () => {
    try {
      setLoading(true);
      if (userRole === 'student') {
        const enrolled = await getEnrolledClasses();
        setClasses(enrolled);
        if (enrolled.length > 0 && !selectedClassId) {
          onClassSelect(enrolled[0].id);
        }
      } else {
        const myClasses = await getMyClasses();
        setClasses(myClasses);
        if (myClasses.length > 0 && !selectedClassId) {
          onClassSelect(myClasses[0].id);
        }
      }
      setError(null);
    } catch (err: any) {
      console.error('Error loading classes:', err);
      setError('Failed to load classes');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-gray-500">
        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-600"></div>
        <span>Loading classes...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-600 text-sm">
        {error}
      </div>
    );
  }

  if (classes.length === 0) {
    return (
      <div className="text-gray-500 text-sm italic">
        {userRole === 'student' ? 'No classes joined yet' : 'No classes created yet'}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <label className="text-sm font-medium text-gray-700">Class:</label>
      <select
        value={selectedClassId || ''}
        onChange={(e) => onClassSelect(Number(e.target.value))}
        className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
      >
        {classes.map((cls) => (
          <option key={cls.id} value={cls.id}>
            {cls.name}
            {userRole === 'student' && 'professor_name' in cls && ` (${cls.professor_name})`}
            {userRole === 'professor' && 'enrolled_count' in cls && ` (${cls.enrolled_count} students)`}
          </option>
        ))}
      </select>
    </div>
  );
};

export default ClassSelector;

