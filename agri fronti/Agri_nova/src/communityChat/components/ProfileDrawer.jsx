import React, { useEffect, useState } from 'react';
import { useChat } from '../context/ChatContext';

const ProfileDrawer = ({ open, onClose }) => {
  const { username, bio, setBio, logout } = useChat();
  const [draftBio, setDraftBio] = useState(bio || '');
  const [status, setStatus] = useState('Available for field chats');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) {
      setDraftBio(bio || '');
    }
  }, [bio, open]);

  const handleSave = () => {
    setSaving(true);
    setTimeout(() => {
      setBio(draftBio.trim() || 'Helping farms thrive every season.');
      setSaving(false);
    }, 200);
  };

  const handleLogout = () => {
    logout();
    if (typeof onClose === 'function') {
      onClose();
    }
  };

  return (
    <aside className={`profile-drawer ${open ? 'profile-drawer--open' : ''}`}>
      <button className="profile-drawer__close" onClick={onClose} aria-label="Close profile drawer">
        ✕
      </button>
      <div className="profile-drawer__header">
        <div className="profile-drawer__avatar">
          {username?.charAt(0)?.toUpperCase()}
        </div>
        <div>
          <p className="profile-drawer__kicker">Logged in as</p>
          <h2>{username}</h2>
          <p className="profile-drawer__status">{status}</p>
        </div>
      </div>

      <div className="profile-drawer__section">
        <p className="profile-drawer__label">Bio</p>
        <textarea
          value={draftBio}
          onChange={(event) => setDraftBio(event.target.value)}
          placeholder="Share a short intro for the community"
        />
        <div className="profile-drawer__actions">
          <button className="ghost-btn" onClick={() => setDraftBio(bio || '')} disabled={saving}>
            Reset
          </button>
          <button className="primary-btn" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving…' : 'Save Bio'}
          </button>
        </div>
      </div>

      <div className="profile-drawer__section">
        <p className="profile-drawer__label">Presence</p>
        <select value={status} onChange={(event) => setStatus(event.target.value)}>
          <option value="Available for field chats">Available for field chats</option>
          <option value="Visiting the crops">Visiting the crops</option>
          <option value="Collecting market updates">Collecting market updates</option>
          <option value="Offline">Offline</option>
        </select>
      </div>

      <div className="profile-drawer__section profile-drawer__section--muted">
        <p className="profile-drawer__label">Security</p>
        <div className="profile-drawer__stat-row">
          <span>Encryption</span>
          <strong>End-to-end</strong>
        </div>
        <div className="profile-drawer__stat-row">
          <span>Member since</span>
          <strong>{new Date().getFullYear()}</strong>
        </div>
      </div>

      <button className="profile-drawer__logout" onClick={handleLogout}>
        Logout
      </button>
    </aside>
  );
};

export default ProfileDrawer;
