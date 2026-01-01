// src/components/PremiumFeatures.tsx
import React from 'react';

const PremiumFeatures: React.FC = () => {
    const features = [
        {
            icon: '🏆',
            title: 'Premium Assets',
            description: 'Exclusive digital collectibles and rare NFTs',
            gradient: 'from-yellow-400 to-orange-500'
        },
        {
            icon: '⚡',
            title: 'Instant Settlement',
            description: 'Lightning-fast cross-chain transactions',
            gradient: 'from-blue-400 to-purple-500'
        },
        {
            icon: '🔐',
            title: 'Bank-Grade Security',
            description: 'Multi-signature smart contract protection',
            gradient: 'from-green-400 to-blue-500'
        },
        {
            icon: '🌍',
            title: 'Global Access',
            description: 'Trade from anywhere, anytime, any device',
            gradient: 'from-purple-400 to-pink-500'
        }
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16 mb-8">
            {features.map((feature, index) => (
                <div
                    key={index}
                    className="group relative overflow-hidden bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-white/20 hover:shadow-2xl transition-all duration-300 hover:transform hover:scale-105"
                    style={{ animationDelay: `${index * 100}ms` }}
                >
                    {/* Gradient overlay on hover */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />

                    <div className="relative z-10 text-center">
                        <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                            {feature.icon}
                        </div>
                        <h3 className="font-bold text-gray-800 mb-2 group-hover:text-gray-900">
                            {feature.title}
                        </h3>
                        <p className="text-gray-600 text-sm leading-relaxed">
                            {feature.description}
                        </p>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default PremiumFeatures;