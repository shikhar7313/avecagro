import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import MainRouter from './MainRouter';

// Suppress Chrome extension async response error
window.addEventListener('unhandledrejection', event => {
  if (event.reason && typeof event.reason.message === 'string' &&
      event.reason.message.includes('listener indicated an asynchronous response')) {
    event.preventDefault();
  }
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MainRouter />
  </React.StrictMode>
);
