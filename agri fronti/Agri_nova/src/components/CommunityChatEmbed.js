import React, { useEffect, useState } from 'react';
import CommunityChatApp from '../communityChat/CommunityChatApp';
import { ChatProvider } from '../communityChat/context/ChatContext';

const CommunityChatEmbed = () => {
  const [sessionKey, setSessionKey] = useState(() => Date.now());
  const [viewportHeight, setViewportHeight] = useState(() =>
    typeof window !== 'undefined' ? window.innerHeight : 900
  );

  useEffect(() => {
    const handleResize = () => {
      setViewportHeight(window.innerHeight);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const embedMinHeight = Math.max(520, viewportHeight - 260);

  return (
    <div className="p-6 w-full flex flex-col gap-4 flex-1 min-h-0">
      <header className="glass-card flex flex-col md:flex-row md:items-center justify-between gap-4 px-6 py-4">
        <div>
          <p className="text-emerald-300 text-sm uppercase tracking-[0.3em]">Community</p>
          <h2 className="text-xl font-semibold text-white">AvecAgro Live Feed</h2>
          <p className="text-gray-300 text-sm">Chat updates stream here from the dedicated community workspace.</p>
        </div>
        <div className="flex gap-3">
          <button
            className="ghost-btn"
            type="button"
            onClick={() => setSessionKey(Date.now())}
          >
            Restart session
          </button>
        </div>
      </header>

      <div
        className="glass-card flex-1 min-h-0 relative flex flex-col overflow-visible"
        style={{ minHeight: embedMinHeight }}
      >
        <div className="embedded-chat w-full h-full flex-1 min-h-0">
          <ChatProvider key={sessionKey}>
            <CommunityChatApp />
          </ChatProvider>
        </div>
      </div>
    </div>
  );
};

export default CommunityChatEmbed;
