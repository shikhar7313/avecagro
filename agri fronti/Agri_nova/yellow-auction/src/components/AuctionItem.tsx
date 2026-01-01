// src/components/AuctionItem.tsx
import React, { useState } from 'react';

interface AuctionItemProps {
    title?: string;
    description?: string;
    imageUrl?: string;
    startingPrice?: number;
    currentBid?: number;
}

const AuctionItem: React.FC<AuctionItemProps> = ({
    title = "Premium Digital Collectible",
    description = "Exclusive limited edition digital asset with verified authenticity and blockchain provenance.",
    imageUrl = "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=400&h=400&fit=crop&crop=center",
    startingPrice = 0.001,
    currentBid = 0
}) => {
    const [imageLoaded, setImageLoaded] = useState(false);
    const [imageError, setImageError] = useState(false);

    return (
        <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/20 overflow-hidden hover:shadow-3xl transition-all duration-300">
            {/* Image Section */}
            <div className="relative h-64 md:h-80 overflow-hidden bg-gradient-to-br from-yellow-100 to-orange-100">
                {!imageError ? (
                    <img
                        src={imageUrl}
                        alt={title}
                        className={`w-full h-full object-cover transition-all duration-500 transform hover:scale-110 ${imageLoaded ? 'opacity-100' : 'opacity-0'
                            }`}
                        onLoad={() => setImageLoaded(true)}
                        onError={() => setImageError(true)}
                    />
                ) : (
                    // Fallback gradient with icon
                    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-yellow-200 via-orange-200 to-red-200">
                        <div className="text-center">
                            <div className="text-6xl mb-4 opacity-60">🏆</div>
                            <div className="text-xl font-bold text-gray-700">Premium Asset</div>
                        </div>
                    </div>
                )}

                {/* Overlay gradient */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent" />

                {/* Badge */}
                <div className="absolute top-4 left-4">
                    <div className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white px-3 py-1 rounded-full text-sm font-bold shadow-lg">
                        🌟 Premium
                    </div>
                </div>

                {/* Price overlay */}
                <div className="absolute bottom-4 right-4">
                    <div className="bg-white/90 backdrop-blur-sm rounded-xl p-3 shadow-lg">
                        <div className="text-xs text-gray-600 font-medium">Starting at</div>
                        <div className="text-lg font-bold text-gray-800">{startingPrice} ETH</div>
                    </div>
                </div>
            </div>

            {/* Content Section */}
            <div className="p-6">
                <h3 className="text-2xl font-bold text-gray-800 mb-3 leading-tight">
                    {title}
                </h3>

                <p className="text-gray-600 mb-6 leading-relaxed">
                    {description}
                </p>

                {/* Features */}
                <div className="grid grid-cols-2 gap-3 mb-6">
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                        <span className="text-green-500">✓</span>
                        <span>Verified Authentic</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                        <span className="text-green-500">✓</span>
                        <span>Blockchain Secured</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                        <span className="text-green-500">✓</span>
                        <span>Limited Edition</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                        <span className="text-green-500">✓</span>
                        <span>Transferable</span>
                    </div>
                </div>

                {/* Rarity and Stats */}
                <div className="border-t border-gray-200 pt-4">
                    <div className="flex justify-between items-center text-sm">
                        <div>
                            <span className="text-gray-600">Rarity:</span>
                            <span className="ml-2 font-bold text-yellow-600">Legendary</span>
                        </div>
                        <div>
                            <span className="text-gray-600">Edition:</span>
                            <span className="ml-2 font-bold text-purple-600">#001/100</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AuctionItem;