import React, { useState } from 'react';

const LoginPage = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError('Enter both username and password.');
      return;
    }
    setError('');
    if (onLogin) {
      onLogin(username.trim());
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-900 via-slate-900 to-black px-4">
      <form
        className="w-full max-w-sm rounded-2xl border border-white/10 bg-white/5 p-8 text-white shadow-2xl backdrop-blur"
        onSubmit={handleSubmit}
      >
        <h1 className="text-2xl font-semibold text-center">Welcome back</h1>
        <p className="mt-2 text-center text-sm text-gray-300">Log in to access the dashboard.</p>

        <label className="mt-6 block text-sm text-gray-200" htmlFor="username">Username</label>
        <input
          id="username"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="mt-1 w-full rounded-xl border border-white/10 bg-black/30 p-3 text-white focus:border-emerald-400 focus:outline-none"
        />

        <label className="mt-4 block text-sm text-gray-200" htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-1 w-full rounded-xl border border-white/10 bg-black/30 p-3 text-white focus:border-emerald-400 focus:outline-none"
        />

        {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}

        <button
          type="submit"
          className="mt-6 w-full rounded-full bg-emerald-500 py-3 text-sm font-semibold text-white hover:bg-emerald-600"
        >
          Log in
        </button>
        <p className="mt-3 text-center text-xs text-gray-400">
          This device will be remembered for faster access next time.
        </p>
      </form>
    </div>
  );
};

export default LoginPage;
