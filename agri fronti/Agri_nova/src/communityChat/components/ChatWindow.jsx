import React from 'react';
import { useChat } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import Composer from './Composer';
import { useAutoScroll } from '../hooks/useAutoScroll';
import { formatRelative } from '../utils/time';

const ChatWindow = ({ onOpenProfile }) => {
  const { rooms, activeRoom, messages, typingUsers } = useChat();
  const activeRoomData = rooms.find((room) => room.id === activeRoom) || rooms[0] || {};
  const scrollRef = useAutoScroll(messages);

  return (
    <section className="chat-window">
      <header className="chat-window__header">
        <div className="chat-window__identity">
          <div className="chat-window__pulse" />
          <div>
            <p className="chat-window__room">#{activeRoomData.name || 'Room'}</p>
            <span className="chat-window__topic">{activeRoomData.description}</span>
          </div>
        </div>
        <div className="chat-window__actions">
          <button className="ghost-btn ghost-btn--round" onClick={onOpenProfile} title="Open profile">
            ⚙️
          </button>
          <button className="ghost-btn ghost-btn--round" title="Pin room">
            📌
          </button>
        </div>
      </header>

      <div className="chat-window__toolbar">
        <span className="chip chip--soft">Live now</span>
        <span className="chip chip--soft">{typingUsers.length ? `${typingUsers.join(', ')} typing…` : 'Community synced'}</span>
        <span className="chip chip--soft" onClick={onOpenProfile} role="button" tabIndex={0}>
          View profile
        </span>
      </div>

      <div className="chat-window__messages" ref={scrollRef}>
        <div className="chat-window__wallpaper" />
        {messages.map((message) => (
          <MessageBubble key={message.id || message.clientId} message={message} />
        ))}
        <TypingIndicator users={typingUsers} />
      </div>

      <Composer />
      <footer className="chat-window__footer">
        Last updated {formatRelative(activeRoomData.updatedAt || new Date().toISOString())}
      </footer>
    </section>
  );
};

export default ChatWindow;
