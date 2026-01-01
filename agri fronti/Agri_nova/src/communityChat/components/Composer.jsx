import React, { useRef, useState } from 'react';
import Picker from 'emoji-picker-react';
import { useChat } from '../context/ChatContext';

const Composer = () => {
  const { sendMessage, setTyping } = useChat();
  const [text, setText] = useState('');
  const [showEmoji, setShowEmoji] = useState(false);
  const [imageData, setImageData] = useState(null);
  const [typingTimeout, setTypingTimeout] = useState(null);
  const fileRef = useRef();

  const emitTyping = () => {
    if (typingTimeout) clearTimeout(typingTimeout);
    setTyping(true);
    const timeout = setTimeout(() => setTyping(false), 1200);
    setTypingTimeout(timeout);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!text.trim() && !imageData) return;
    const clientId = window.crypto?.randomUUID ? window.crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
    sendMessage({ text: text.trim(), image: imageData, clientId });
    setText('');
    setImageData(null);
    setShowEmoji(false);
    setTyping(false);
  };

  // Support both emoji-picker-react v3 (event, data) and v4 (data, event)
  const handleEmoji = (firstArg, secondArg) => {
    const data = firstArg?.emoji ? firstArg : secondArg?.emoji ? secondArg : null;
    if (!data?.emoji) return;
    setText((prev) => `${prev}${data.emoji}`);
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageData(reader.result);
    reader.readAsDataURL(file);
  };

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <div className="composer__glass">
        <div className="composer__actions">
          <button type="button" onClick={() => setShowEmoji((prev) => !prev)} aria-label="Toggle emoji picker">
            😊
          </button>
          <button type="button" onClick={() => fileRef.current?.click()} aria-label="Attach image">
            📎
          </button>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
        </div>
        <input
          className="composer__input"
          type="text"
          placeholder="Share an update with the community"
          value={text}
          onChange={(event) => {
            setText(event.target.value);
            emitTyping();
          }}
        />
        <div className="composer__send-wrap">
          <button className="composer__send" type="submit" disabled={!text.trim() && !imageData}>
            <span className="composer__send-label">Send</span>
            <span className="composer__send-icon">➤</span>
          </button>
        </div>
      </div>
      {showEmoji && (
        <div className="composer__emoji-picker">
          <Picker onEmojiClick={handleEmoji} width="100%" height={300} theme="dark" />
        </div>
      )}
      {imageData && (
        <div className="composer__preview">
          <img src={imageData} alt="preview" />
          <button type="button" onClick={() => setImageData(null)}>
            ✕
          </button>
        </div>
      )}
      <p className="composer__hint">Press Enter to send • Shift + Enter for newline</p>
    </form>
  );
};

export default Composer;
