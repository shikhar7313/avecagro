// src/components/AuctionStats.tsx
import React from 'react';

interface AuctionStatsProps {
    totalBids: number;
    highestBid: number;
    timeLeft: number;
    participants: number;
}

const AuctionStats: React.FC<AuctionStatsProps> = ({
    totalBids,
    highestBid,
    timeLeft,
    participants
}) => {
    const stats = [
        {
            label: 'Total Bids',
            value: totalBids.toString(),
            icon: '📊',
            color: 'text-blue-600'
        },
        {
            label: 'Highest Bid',
            value: `${highestBid.toFixed(4)} ETH`,
            icon: '💎',
            color: 'text-yellow-600'
        },
        {
            label: 'Participants',
            value: participants.toString(),
            icon: '👥',
            color: 'text-green-600'
        },
        {
            label: 'Status',
            value: timeLeft > 0 ? 'Active' : 'Ended',
            icon: timeLeft > 0 ? '🟢' : '🔴',
            color: timeLeft > 0 ? 'text-green-600' : 'text-red-600'
        }
    ];

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {stats.map((stat, index) => (
                <div
                    key={index}
                    className="bg-white/90 backdrop-blur-sm rounded-xl p-4 text-center shadow-lg border border-white/20 hover:shadow-xl transition-all duration-200"
                >
                    <div className="text-2xl mb-2">{stat.icon}</div>
                    <div className={`text-xl font-bold ${stat.color} mb-1`}>
                        {stat.value}
                    </div>
                    <div className="text-sm text-gray-600 font-medium">
                        {stat.label}
                    </div>
                </div>
            ))}
        </div>
    );
};

export default AuctionStats;