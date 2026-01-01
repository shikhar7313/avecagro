import React, { useMemo, useState } from 'react';
import { formatTime } from '../utils/time';
import { useChat } from '../context/ChatContext';

const Sidebar = ({ rooms, activeRoom, onSelectRoom, onlineUsers, allMessages, onOpenProfile }) => {
  const { username, bio } = useChat();
  const profileInitial = username?.charAt(0)?.toUpperCase() || '?';
  const [query, setQuery] = useState('');
  const filteredRooms = useMemo(() => {
    if (!query.trim()) return rooms;
    const lower = query.toLowerCase();
    return rooms.filter((room) => room.name.toLowerCase().includes(lower) || room.description?.toLowerCase().includes(lower));
  }, [rooms, query]);
  const initial = (value) => value?.charAt(0)?.toUpperCase() || 'A';

  const renderPreview = (roomId) => {
    const messages = allMessages?.[roomId] || [];
    if (!messages.length) return 'Say hi to everyone 👋';
    const last = messages[messages.length - 1];
    return `${last.sender === username ? 'You' : last.sender}: ${last.text || '📷 Image'}`;
  };

  const unreadCount = (roomId) => {
    const messages = allMessages?.[roomId] || [];
    return messages.filter((msg) => msg.sender !== username && !msg.status?.seenBy?.includes(username)).length;
  };

  const totalUnread = useMemo(
    () => rooms.reduce((sum, room) => sum + unreadCount(room.id), 0),
    [rooms, allMessages, username]
  );

  return (
    <aside className="sidebar">
      <div className="sidebar__hero" onClick={onOpenProfile} role="button" tabIndex={0}>
        <div>
          <p className="sidebar__kicker">Welcome back</p>
          <h2>{username || 'Guest'}</h2>
          <p className="sidebar__bio">{bio || 'Helping farms thrive every season.'}</p>
        </div>
        <div className="sidebar__hero-avatar">{profileInitial}</div>
      </div>

      <div className="sidebar__stats">
        <div className="stat-card">
          <span>Rooms</span>
          <strong>{rooms.length}</strong>
          <p>for collaboration</p>
        </div>
        <div className="stat-card">
          <span>Unread</span>
          <strong>{totalUnread}</strong>
          <p>messages waiting</p>
        </div>
      </div>

      <div className="sidebar__search">
        <input
          type="search"
          placeholder="Search rooms or topics"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button onClick={onOpenProfile} className="ghost-btn ghost-btn--tiny">Profile</button>
      </div>

      <div className="sidebar__section">
        <p className="sidebar__section-title">Community Rooms</p>
        <div className="room-list">
          {filteredRooms.map((room) => (
            <button
              key={room.id}
              className={`room-item ${room.id === activeRoom ? 'room-item--active' : ''}`}
              onClick={() => onSelectRoom(room.id)}
            >
              <div>
                <p className="room-item__name">#{room.name}</p>
                <p className="room-item__preview">{renderPreview(room.id)}</p>
              </div>
              <div className="room-item__meta">
                <span>
                  {formatTime(
                    (allMessages?.[room.id] || []).slice(-1)[0]?.timestamp ||
                      room.updatedAt ||
                      room.createdAt
                  )}
                </span>
                {unreadCount(room.id) > 0 && <span className="room-item__badge">{unreadCount(room.id)}</span>}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar__section">
        <p className="sidebar__section-title">Online ({onlineUsers.length})</p>
        <div className="avatar-strip">
          {onlineUsers.map((user) => (
            <div key={user.socketId} className="avatar-strip__item">
              <div className="avatar-strip__avatar">{initial(user.username)}</div>
              <span>{user.username}</span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
