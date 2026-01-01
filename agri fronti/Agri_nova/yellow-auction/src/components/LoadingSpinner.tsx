// src/components/LoadingSpinner.tsx
import React from 'react';

interface LoadingSpinnerProps {
    size?: 'sm' | 'md' | 'lg';
    color?: 'yellow' | 'white' | 'gray';
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
    size = 'md',
    color = 'yellow'
}) => {
    const sizeClasses = {
        sm: 'w-4 h-4',
        md: 'w-6 h-6',
        lg: 'w-8 h-8'
    };

    const colorClasses = {
        yellow: 'border-yellow-500 border-t-transparent',
        white: 'border-white border-t-transparent',
        gray: 'border-gray-500 border-t-transparent'
    };

    return (
        <div
            className={`
        ${sizeClasses[size]} 
        ${colorClasses[color]} 
        border-2 rounded-full animate-spin
      `}
        />
    );
};

export default LoadingSpinner;