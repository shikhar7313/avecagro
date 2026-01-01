import React, { useCallback, useEffect, useMemo, useState } from 'react';

const RAW_URL = process.env.REACT_APP_VITE_LANDING_URL || 'http://localhost:6001';
const landingUrl = RAW_URL.endsWith('/') ? RAW_URL.slice(0, -1) : RAW_URL;

const LandingPage = ({ onContinue }) => {
  const [iframeReady, setIframeReady] = useState(false);
  const [isReachable, setIsReachable] = useState(true);
  const [checking, setChecking] = useState(true);

  const landingOrigin = useMemo(() => {
    try {
      return new URL(landingUrl).origin;
    } catch (error) {
      console.error('Invalid landing URL for Vite server:', error);
      return null;
    }
  }, []);

  const handleContinue = useCallback((payload = {}) => {
    if (onContinue) onContinue(payload);
  }, [onContinue]);

  const probeLandingServer = useCallback(() => {
    if (!landingOrigin) {
      setChecking(false);
      setIsReachable(false);
      return undefined;
    }

    let cancelled = false;
    const controller = new AbortController();
    setChecking(true);

    fetch(landingUrl, { mode: 'no-cors', signal: controller.signal })
      .then(() => {
        if (!cancelled) {
          setIsReachable(true);
          setChecking(false);
        }
      })
      .catch((error) => {
        console.warn('Vite landing server is offline, showing fallback instructions.', error);
        if (!cancelled) {
          setIsReachable(false);
          setChecking(false);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  useEffect(() => probeLandingServer(), [probeLandingServer]);

  useEffect(() => {
    if (!landingOrigin) return undefined;

    const handler = (event) => {
      if (event.origin !== landingOrigin) return;
      if (event.data && event.data.type === 'agri-lp-continue') {
        handleContinue(event.data.payload || {});
      }
    };

    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [landingOrigin, handleContinue]);

  if (!landingOrigin || !isReachable) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-black px-6 text-center text-white">
        <div className="max-w-xl rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur">
          <h1 className="text-2xl font-semibold">Vite landing server is not running</h1>
          <p className="mt-4 text-sm text-gray-300">
            Start the landing project located at
          </p>
          <pre className="mt-3 w-full overflow-auto rounded-xl bg-black/40 p-3 text-left text-xs text-emerald-200">
D:\Gear\agri fronti\ansh\agri fronti\Agri_nova\Landing page\New folder
npm install
npm run dev -- --port 6001
          </pre>
          <p className="mt-2 text-xs text-gray-400">Expected URL: {landingUrl || 'invalid URL'}</p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <button
              type="button"
              className="rounded-full bg-white/20 px-4 py-2 text-sm font-semibold text-white hover:bg-white/30"
              onClick={probeLandingServer}
            >
              {checking ? 'Checking…' : 'Retry connection'}
            </button>
            <button
              type="button"
              className="rounded-full bg-emerald-500 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-600"
              onClick={() => handleContinue()}
            >
              Continue without landing
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-screen w-full bg-black">
      {!iframeReady && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-black/80 text-center text-white">
          <p className="text-lg font-semibold">Launching landing experience…</p>
          <p className="text-sm text-gray-300">Serving content from {landingUrl}</p>
        </div>
      )}

      <iframe
        title="AgriNova landing"
        src={landingUrl}
        className="h-full w-full border-0"
        sandbox="allow-same-origin allow-scripts allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox"
        allow="clipboard-write; accelerometer; autoplay; gyroscope"
        onLoad={() => setIframeReady(true)}
      />

    </div>
  );
};

export default LandingPage;
