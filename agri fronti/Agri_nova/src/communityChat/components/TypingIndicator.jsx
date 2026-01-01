import React from 'react';

const TypingIndicator = ({ users = [] }) => {
  if (!users.length) return null;
  return (
    <div className="typing-indicator">
      <span>{users.join(', ')} typing</span>
      <div className="typing-indicator__dots">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
};

export default TypingIndicator;
