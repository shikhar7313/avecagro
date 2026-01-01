// src/components/NotificationToast.tsx
import React, { useState, useEffect } from 'react';

interface NotificationProps {
    message: string;
    type: 'success' | 'error' | 'info' | 'warning';
    isVisible: boolean;
    onClose: () => void;
    duration?: number;
}

const NotificationToast: React.FC<NotificationProps> = ({
    message,
    type,
    isVisible,
    onClose,
    duration = 4000
}) => {
    useEffect(() => {
        if (isVisible && duration > 0) {
            const timer = setTimeout(() => {
                onClose();
            }, duration);

            return () => clearTimeout(timer);
        }
    }, [isVisible, duration, onClose]);

    if (!isVisible) return null;

    const typeConfig = {
        success: {
            icon: '✅',
            bgColor: 'bg-green-500',
            textColor: 'text-white',
            borderColor: 'border-green-600'
        },
        error: {
            icon: '❌',
            bgColor: 'bg-red-500',
            textColor: 'text-white',
            borderColor: 'border-red-600'
        },
        warning: {
            icon: '⚠️',
            bgColor: 'bg-yellow-500',
            textColor: 'text-black',
            borderColor: 'border-yellow-600'
        },
        info: {
            icon: 'ℹ️',
            bgColor: 'bg-blue-500',
            textColor: 'text-white',
            borderColor: 'border-blue-600'
        }
    };

    const config = typeConfig[type];

    return (
        <div className="fixed top-4 right-4 z-50 animate-slide-up">
            <div className={`
        ${config.bgColor} ${config.textColor} ${config.borderColor}
        px-6 py-4 rounded-2xl shadow-2xl border-2 backdrop-blur-sm
        flex items-center gap-3 min-w-[300px] max-w-md
        transform transition-all duration-300 hover:scale-105
      `}>
                <div className="text-xl">{config.icon}</div>
                <div className="flex-1 font-medium">{message}</div>
                <button
                    onClick={onClose}
                    className={`
            ml-2 text-lg font-bold opacity-70 hover:opacity-100 
            transition-opacity duration-200 ${config.textColor}
          `}
                >
                    ×
                </button>
            </div>
        </div>
    );
};

// Hook for managing notifications
export const useNotification = () => {
    const [notifications, setNotifications] = useState<Array<{
        id: string;
        message: string;
        type: 'success' | 'error' | 'info' | 'warning';
    }>>([]);

    const showNotification = (message: string, type: 'success' | 'error' | 'info' | 'warning') => {
        const id = Date.now().toString();
        setNotifications(prev => [...prev, { id, message, type }]);
    };

    const hideNotification = (id: string) => {
        setNotifications(prev => prev.filter(n => n.id !== id));
    };

    const NotificationContainer = () => (
        <>
            {notifications.map((notification) => (
                <NotificationToast
                    key={notification.id}
                    message={notification.message}
                    type={notification.type}
                    isVisible={true}
                    onClose={() => hideNotification(notification.id)}
                />
            ))}
        </>
    );

    return { showNotification, NotificationContainer };
};

export default NotificationToast;