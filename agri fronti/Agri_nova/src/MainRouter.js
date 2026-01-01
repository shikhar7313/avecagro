import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import AppWrapper from './components/AppWrapper';
import Chatbot from './components/Chatbot';
import { fetchDeviceIp, getOrCreateDeviceId, getRememberedUserForDevice, rememberDeviceForUser } from './utils/deviceUtils';

// Simple auth hook using localStorage
function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    localStorage.getItem('isLoggedIn') === 'true'
  );
  const [deviceIp, setDeviceIp] = useState(null);
  const [deviceId] = useState(() => getOrCreateDeviceId());
  const [isRestoringSession, setIsRestoringSession] = useState(true);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const rememberedUser = getRememberedUserForDevice(deviceId);
    if (rememberedUser) {
      setIsAuthenticated((prev) => {
        if (!localStorage.getItem('username')) {
          localStorage.setItem('username', rememberedUser);
        }
        localStorage.setItem('isLoggedIn', 'true');
        return true;
      });
    }

    const resolveDevice = async () => {
      try {
        const ip = await fetchDeviceIp(controller.signal);
        if (!isMounted) return;
        setDeviceIp(ip);
      } catch (error) {
        console.warn('Failed to resolve device IP for auto-login', error);
      } finally {
        if (isMounted) {
          setIsRestoringSession(false);
        }
      }
    };

    resolveDevice();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [deviceId]);

  const login = useCallback((username) => {
    setIsAuthenticated(true);
    localStorage.setItem('isLoggedIn', 'true');
    localStorage.setItem('username', username);
    rememberDeviceForUser(username, { deviceId, deviceIp });
  }, [deviceId, deviceIp]);

  const logout = useCallback(() => {
    setIsAuthenticated(false);
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('username');
  }, []);

  return { isAuthenticated, login, logout, deviceIp, isRestoringSession };
}

const MainRouter = () => {
  const auth = useAuth();
  const { logout } = auth;
  const [hasVisitedLanding, setHasVisitedLanding] = useState(
    sessionStorage.getItem('hasVisitedLanding') === 'true'
  );

  // Log out only on real tab/browser close (not reload/navigation)
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      // If the document is being hidden and not persisted (not a reload/navigation)
      if (document.visibilityState === 'hidden' && !e.persisted) {
        logout();
        sessionStorage.removeItem('hasVisitedLanding');
      }
    };
    window.addEventListener('pagehide', handleBeforeUnload);
    return () => window.removeEventListener('pagehide', handleBeforeUnload);
  }, [logout]);

  if (auth.isRestoringSession) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
        <p>Reconnecting to your trusted device...</p>
      </div>
    );
  }

  const LandingRoute = () => {
    const navigate = useNavigate();
    const handleContinue = (payload = {}) => {
      setHasVisitedLanding(true);
      sessionStorage.setItem('hasVisitedLanding', 'true');

      if (!auth.isAuthenticated) {
        if (payload.username) {
          auth.login(payload.username);
          navigate('/dashboard');
          return;
        }
        navigate('/login');
        return;
      }

      navigate('/dashboard');
    };
    return <LandingPage onContinue={handleContinue} />;
  };

  return (
    <Router>
      <Routes>
        <Route path="/landing" element={<LandingRoute />} />
        <Route
          path="/login"
          element={
            <LoginPage onLogin={auth.login} />
          }
        />
        <Route
          path="/dashboard"
          element={
            auth.isAuthenticated ? (
              <AppWrapper onLogout={auth.logout} />
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
        <Route
          path="/chatbot"
          element={
            auth.isAuthenticated ? (
              <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-green-900">
                <Chatbot />
              </div>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
        <Route
          path="/"
          element={
            !hasVisitedLanding ? (
              <Navigate to="/landing" replace />
            ) : auth.isAuthenticated ? (
              <Navigate to="/dashboard" replace />
            ) : (
              <Navigate to="/landing" replace />
            )
          }
        />
        {/* Fallback: redirect unknown routes to landing */}
        <Route path="*" element={<Navigate to="/landing" replace />} />
      </Routes>
    </Router>
  );
};

export default MainRouter;
