import React from 'react';
import ChatPage from './pages/ChatPage';
import UsernameModal from './components/UsernameModal';
import { useChat } from './context/ChatContext';
import { useNotifications } from './hooks/useNotifications';
import { usePushRegistration } from './hooks/usePushRegistration';
import './styles/chat.css';

const CommunityChatApp = () => {
  const { username, setUsername, lastIncomingMessage } = useChat();

  useNotifications(
    lastIncomingMessage ? `${lastIncomingMessage.sender} messaged you` : '',
    lastIncomingMessage?.text || 'New message'
  );
  usePushRegistration(Boolean(username));

  return (
    <div className="app-shell">
      {!username && <UsernameModal onSubmit={setUsername} />}
      <ChatPage />
    </div>
  );
};

export default CommunityChatApp;
