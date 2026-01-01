import React, { useState } from 'react';
import Sidebar from '../components/Sidebar';
import ChatWindow from '../components/ChatWindow';
import ProfileDrawer from '../components/ProfileDrawer';
import { useChat } from '../context/ChatContext';

const ChatPage = () => {
  const { rooms, activeRoom, setActiveRoom, allMessages, onlineUsers } = useChat();
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <div className="chat-layout">
      <Sidebar
        rooms={rooms}
        activeRoom={activeRoom}
        onSelectRoom={setActiveRoom}
        onlineUsers={onlineUsers}
        allMessages={allMessages}
        onOpenProfile={() => setDrawerOpen(true)}
      />
      <ChatWindow onOpenProfile={() => setDrawerOpen(true)} />
      <ProfileDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </div>
  );
};

export default ChatPage;
