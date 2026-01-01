import React, { useState } from 'react';
import '../styles/modal.css';

const UsernameModal = ({ onSubmit }) => {
  const [value, setValue] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!value.trim()) return;
    onSubmit(value.trim());
  };

  return (
    <div className="modal-backdrop">
      <form className="modal" onSubmit={handleSubmit}>
        <h2>Choose a username</h2>
        <p>Your username identifies this browser tab.</p>
        <input
          type="text"
          placeholder="e.g. FarmHero73"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          autoFocus
        />
        <button type="submit">Continue</button>
      </form>
    </div>
  );
};

export default UsernameModal;
