import React from 'react';
import App from '../App';

const AppWrapper = ({ onLogout }) => {
    return <App onLogout={onLogout} />;
};

export default AppWrapper;