/**
 * Main App component with routing
 */
import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { auth } from './services/firebase';
import { onAuthStateChanged } from 'firebase/auth';
import Login from './components/Login';
import Student from './pages/Student';
import Professor from './pages/Professor';
import ClassManagement from './pages/ClassManagement';
import ChangePasswordModal from './components/ChangePasswordModal';

function App() {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showPasswordModal, setShowPasswordModal] = useState(false);

  useEffect(() => {
    // Check localStorage first for professor or student data
    const checkAuth = () => {
      const userData = localStorage.getItem('user');
      if (userData) {
        setUser(JSON.parse(userData));
      } else {
        setUser(null);
      }
      setLoading(false);
    };

    // Listen for Firebase auth state changes (students)
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      if (firebaseUser) {
        const userData = localStorage.getItem('user');
        if (userData) {
          setUser(JSON.parse(userData));
        }
      } else {
        // Only clear if no professor token exists
        if (!localStorage.getItem('professorToken')) {
          setUser(null);
          localStorage.removeItem('user');
        }
      }
      setLoading(false);
    });

    // Listen for custom logout event
    const handleLogoutEvent = () => {
      setUser(null);
    };
    
    window.addEventListener('logout', handleLogoutEvent);

    // Initial check
    checkAuth();

    return () => {
      unsubscribe();
      window.removeEventListener('logout', handleLogoutEvent);
    };
  }, []);

  const handleLoginSuccess = (userData: any) => {
    setUser(userData);
    // Check if professor must change password
    if (userData.role === 'professor' && userData.must_change_password) {
      setShowPasswordModal(true);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route
          path="/login"
          element={
            user ? (
              <Navigate to={user.role === 'professor' ? '/professor' : '/student'} />
            ) : (
              <Login onLoginSuccess={handleLoginSuccess} />
            )
          }
        />
        <Route
          path="/student"
          element={
            user && user.role === 'student' ? (
              <Student />
            ) : (
              <Navigate to="/login" />
            )
          }
        />
        <Route
          path="/professor"
          element={
            user && user.role === 'professor' ? (
              <Professor />
            ) : (
              <Navigate to="/login" />
            )
          }
        />
        <Route
          path="/professor/classes"
          element={
            user && user.role === 'professor' ? (
              <ClassManagement />
            ) : (
              <Navigate to="/login" />
            )
          }
        />
        <Route
          path="/"
          element={
            user ? (
              <Navigate to={user.role === 'professor' ? '/professor' : '/student'} />
            ) : (
              <Navigate to="/login" />
            )
          }
        />
      </Routes>
      
      {/* Change Password Modal */}
      {showPasswordModal && user?.role === 'professor' && (
        <ChangePasswordModal
          isOpen={showPasswordModal}
          onClose={() => setShowPasswordModal(false)}
          onSuccess={() => {
            setShowPasswordModal(false);
            // Update user data to reflect password change
            const updatedUser = { ...user, must_change_password: false };
            setUser(updatedUser);
            localStorage.setItem('user', JSON.stringify(updatedUser));
          }}
          required={user.must_change_password}
        />
      )}
    </Router>
  );
}

export default App;

