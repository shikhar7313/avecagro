import React from 'react';
import { formatTime } from '../utils/time';
import { getReceiptClass, getReceiptIcon } from '../utils/readReceipts';
import { useChat } from '../context/ChatContext';

const MessageBubble = ({ message }) => {
  const { username } = useChat();
  const isOwner = message.sender === username;
  const receiptIcon = getReceiptIcon({ ...message.status, username });
  const receiptClass = getReceiptClass({ ...message.status, username });

  return (
    <div className={`bubble ${isOwner ? 'bubble--own' : ''}`}>
      {!isOwner && <p className="bubble__sender">{message.sender}</p>}
      {message.image && (
        <img src={message.image} alt="shared" className="bubble__image" />
      )}
      {message.text && <p className="bubble__text">{message.text}</p>}
      <div className="bubble__meta">
        <span>{formatTime(message.timestamp)}</span>
        {isOwner && <span className={receiptClass}>{receiptIcon}</span>}
      </div>
    </div>
  );
};

export default MessageBubble;
