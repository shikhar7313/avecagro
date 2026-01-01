import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { io } from 'socket.io-client';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { fetchRooms } from '../utils/api';

const ChatContext = createContext();
const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'http://localhost:4000';

const generateGuestHandle = () => {
  if (typeof window === 'undefined') return 'FieldGuest';
  const candidate =
    window.localStorage.getItem('agrinova.user') ||
    window.localStorage.getItem('agrinova.username') ||
    window.localStorage.getItem('agrinova.fullname');
  if (candidate) {
    try {
      const parsed = JSON.parse(candidate);
      if (typeof parsed === 'string' && parsed.trim()) {
        return parsed.trim();
      }
    } catch (error) {
      if (candidate.trim()) {
        return candidate.trim();
      }
    }
  }
  return `FieldGuest-${Math.floor(100 + Math.random() * 900)}`;
};

const toArray = (value) => {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (value instanceof Set) return Array.from(value);
  if (typeof value === 'object' && Array.isArray(value.data)) {
    return value.data;
  }
  try {
    return Array.from(value);
  } catch (error) {
    return [];
  }
};

const normalizeMessage = (message) => ({
  ...message,
  status: {
    sentAt: message?.status?.sentAt || Date.now(),
    deliveredTo: toArray(message?.status?.deliveredTo),
    seenBy: toArray(message?.status?.seenBy)
  }
});

export const ChatProvider = ({ children }) => {
  const [username, setUsername] = useLocalStorage('agrichat.username', generateGuestHandle);
  const [bio, setBio] = useLocalStorage('agrichat.bio', 'Helping farms thrive every season.');
  const [socket, setSocket] = useState(null);
  const [rooms, setRooms] = useState([]);
  const [activeRoom, setActiveRoom] = useState('general');
  const [messagesByRoom, setMessagesByRoom] = useState({});
  const [typingUsers, setTypingUsers] = useState({});
  const [onlineUsers, setOnlineUsers] = useState([]);
  const [lastIncomingMessage, setLastIncomingMessage] = useState(null);

  useEffect(() => {
    fetchRooms()
      .then((data) => setRooms(data.rooms || []))
      .catch(() => setRooms([]));
  }, []);

  useEffect(() => {
    if (!username) return;
    const instance = io(SOCKET_URL, { query: { username } });
    setSocket(instance);

    instance.on('connect', () => {
      console.info('socket connected');
    });

    instance.on('room-history', ({ roomId, messages }) => {
      setMessagesByRoom((prev) => ({
        ...prev,
        [roomId]: messages.map(normalizeMessage)
      }));
    });

    instance.on('message', (message) => {
      const normalized = normalizeMessage(message);
      setMessagesByRoom((prev) => {
        const list = prev[normalized.roomId] || [];
        const idx = list.findIndex((item) => item.id === normalized.id || (normalized.clientId && item.clientId === normalized.clientId));
        const next = idx >= 0 ? [...list.slice(0, idx), normalized, ...list.slice(idx + 1)] : [...list, normalized];
        return { ...prev, [normalized.roomId]: next };
      });
      if (normalized.sender !== username) {
        setLastIncomingMessage(normalized);
        const payload = { roomId: normalized.roomId, messageId: normalized.id || normalized.clientId };
        const isActiveRoom = normalized.roomId === activeRoom && !document.hidden;
        instance.emit(isActiveRoom ? 'message-seen' : 'message-delivered', payload);
      }
    });

    instance.on('message-status', (message) => {
      const normalized = normalizeMessage(message);
      setMessagesByRoom((prev) => {
        const list = prev[normalized.roomId] || [];
        const idx = list.findIndex((item) => item.id === normalized.id || (normalized.clientId && item.clientId === normalized.clientId));
        if (idx === -1) return prev;
        const next = [...list];
        next[idx] = normalized;
        return { ...prev, [normalized.roomId]: next };
      });
    });

    instance.on('typing', ({ roomId, username: typingUser, isTyping }) => {
      setTypingUsers((prev) => {
        const users = new Set(prev[roomId] || []);
        if (isTyping) {
          users.add(typingUser);
        } else {
          users.delete(typingUser);
        }
        return { ...prev, [roomId]: Array.from(users) };
      });
    });

    instance.on('user-connected', ({ users }) => setOnlineUsers(users));
    instance.on('user-disconnected', ({ users }) => setOnlineUsers(users));

    return () => instance.disconnect();
  }, [username, activeRoom]);

  useEffect(() => {
    if (!socket || !activeRoom) return;
    socket.emit('join-room', activeRoom);
  }, [socket, activeRoom]);

  const sendMessage = ({ text, image, clientId }) => {
    if (!socket || !activeRoom) return;
    socket.emit('chat-message', { roomId: activeRoom, text, image, clientId });
  };

  const setTyping = (isTyping) => {
    if (!socket || !activeRoom) return;
    socket.emit('typing', { roomId: activeRoom, isTyping });
  };

  const logout = () => {
    setUsername('');
  };

  const value = useMemo(() => ({
    username,
    setUsername,
    bio,
    setBio,
    logout,
    socket,
    rooms,
    activeRoom,
    setActiveRoom,
    messages: messagesByRoom[activeRoom] || [],
    allMessages: messagesByRoom,
    sendMessage,
    typingUsers: typingUsers[activeRoom] || [],
    setTyping,
    onlineUsers,
    lastIncomingMessage
  }), [username, rooms, activeRoom, messagesByRoom, typingUsers, onlineUsers, lastIncomingMessage, bio]);

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};

export const useChat = () => useContext(ChatContext);
