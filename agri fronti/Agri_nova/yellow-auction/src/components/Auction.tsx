// src/components/Auction.tsx
import React, { useState, useEffect } from "react";
import { useAccount } from "wagmi";
import Payment from "./Payment";
import AuctionStats from "./AuctionStats";
import PremiumFeatures from "./PremiumFeatures";
import AuctionItem from "./AuctionItem";

interface Bid {
  bidder: string;
  amount: string;
  chain: string;
  timestamp: number;
}

interface AuctionProps {
  showNotification: (message: string, type: 'success' | 'error' | 'info' | 'warning') => void;
}

export default function Auction({ showNotification }: AuctionProps) {
  const { address } = useAccount();
  const [bids, setBids] = useState<Bid[]>([]);
  const [amount, setAmount] = useState("0.01");
  const [chain, setChain] = useState<"mainnet" | "sepolia">("sepolia");
  const [timeLeft, setTimeLeft] = useState(3600); // 1 hour auction
  const [isActive, setIsActive] = useState(true);
  const [walletNotificationShown, setWalletNotificationShown] = useState(false);

  // Countdown timer and wallet check
  useEffect(() => {
    if (!address && !walletNotificationShown) {
      showNotification("Please connect your wallet to participate.", "info");
      setWalletNotificationShown(true);
    }

    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          if (isActive) { // Only show notification once
            showNotification("The auction has ended!", "warning");
          }
          setIsActive(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [address, isActive, showNotification, walletNotificationShown]);

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const onSuccess = () => {
    if (address) {
      const newBid = {
        bidder: address,
        amount,
        chain,
        timestamp: Date.now()
      };
      setBids(prev => [...prev, newBid]);
      showNotification(`Successfully placed bid of ${amount} ETH!`, "success");
    }
  };

  const onError = (message: string) => {
    showNotification(message, "error");
  };

  const highestBid = bids.length > 0
    ? Math.max(...bids.map(bid => parseFloat(bid.amount)))
    : 0;

  const uniqueParticipants = new Set(bids.map(bid => bid.bidder)).size;

  const getChainIcon = (chainName: string) => {
    return chainName === "mainnet" ? "🌐" : "🧪";
  };

  const shortenAddress = (addr: string) => {
    return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
  };

  return (
    <div className="w-full animate-slide-up">
      {/* Auction Statistics - Centered */}
      <div className="flex justify-center mb-12">
        <AuctionStats
          totalBids={bids.length}
          highestBid={highestBid}
          timeLeft={timeLeft}
          participants={uniqueParticipants}
        />
      </div>

      {/* Main Auction Layout */}
      <div className="max-w-7xl mx-auto">
        <div className="grid xl:grid-cols-5 gap-8 lg:gap-12">
          {/* Left Column: Auction Item Showcase */}
          <div className="xl:col-span-2 order-2 xl:order-1">
            <div className="sticky top-8">
              <AuctionItem
                currentBid={highestBid}
                startingPrice={0.001}
              />
            </div>
          </div>

          {/* Right Column: Bidding Interface */}
          <div className="xl:col-span-3 order-1 xl:order-2">
            {/* Yellow-themed Auction Card */}
            <div className="bg-black/95 backdrop-blur-md rounded-3xl shadow-2xl border-2 border-yellow-400/60 shadow-yellow-500/20 overflow-hidden hover:shadow-yellow-400/30 transition-all duration-500">
              {/* Yellow-themed Header with Timer */}
              <div className="bg-gradient-to-r from-yellow-400 via-amber-500 to-yellow-600 p-6 sm:p-8 text-black relative overflow-hidden">
                {/* Hexagonal pattern background */}
                <div className="absolute inset-0 opacity-10">
                  <div className="absolute top-4 left-4 w-12 h-12 border-2 border-black transform rotate-45"></div>
                  <div className="absolute top-8 right-8 w-8 h-8 border-2 border-black transform rotate-12"></div>
                  <div className="absolute bottom-4 left-12 w-10 h-10 border-2 border-black transform -rotate-12"></div>
                  <div className="absolute bottom-8 right-4 w-6 h-6 border-2 border-black transform rotate-45"></div>
                </div>

                <div className="relative z-10">
                  <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                    <div>
                      <h2 className="text-2xl sm:text-3xl font-black mb-2 flex items-center gap-3 text-black">
                        � YELLOW AUCTION
                        {isActive && <span className="inline-block w-4 h-4 bg-black rounded-full animate-pulse"></span>}
                      </h2>
                      <p className="text-black/80 text-sm sm:text-base font-semibold">Exclusive Digital Asset • Limited Time Only</p>
                    </div>
                    <div className="text-center sm:text-right bg-black/90 rounded-xl p-4 backdrop-blur border-2 border-black">
                      <div className={`text-xl sm:text-2xl font-mono font-black text-yellow-400 ${timeLeft < 300 ? 'animate-pulse text-red-400' : ''}`}>
                        {formatTime(timeLeft)}
                      </div>
                      <div className="text-xs sm:text-sm text-yellow-200 font-bold">
                        {isActive ? 'TIME LEFT' : 'ENDED'}
                      </div>
                    </div>
                  </div>

                  {/* Current Highest Bid - Yellow Style */}
                  <div className="mt-6 p-4 bg-black/90 rounded-2xl backdrop-blur border-2 border-black shadow-lg">
                    <div className="flex justify-between items-center">
                      <span className="text-yellow-400 font-black text-lg">🏆 HIGHEST BID:</span>
                      <div className="text-right">
                        <div className="text-2xl font-black text-yellow-300">{highestBid.toFixed(4)} ETH</div>
                        <div className="text-sm text-yellow-200 font-semibold">${(highestBid * 2500).toLocaleString()} USD</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Yellow-themed Bidding Interface */}
              <div className="p-6 sm:p-8 bg-gradient-to-br from-black/50 to-gray-900/80">
                {/* Enhanced Bidding Form */}
                <div className="grid lg:grid-cols-2 gap-8">
                  {/* Left Column - Bid Form */}
                  <div className="space-y-6">
                    <div className="space-y-4">
                      <label className="block text-lg font-black text-yellow-400 mb-3 flex items-center gap-2">
                        💰 YOUR BID AMOUNT
                      </label>
                      <div className="relative group">
                        <input
                          type="number"
                          step="0.001"
                          min="0.001"
                          value={amount}
                          onChange={(e) => setAmount(e.target.value)}
                          className="w-full text-2xl font-black text-center p-6 border-2 border-yellow-400/60 rounded-2xl focus:border-yellow-400 focus:ring-4 focus:ring-yellow-400/30 transition-all duration-300 bg-gradient-to-r from-yellow-50 to-amber-50 hover:from-yellow-100 hover:to-amber-100 shadow-inner group-hover:shadow-lg text-black"
                          placeholder="0.001"
                          disabled={!isActive}
                        />
                        <div className="absolute right-6 top-1/2 transform -translate-y-1/2 text-black font-black text-lg">
                          ETH
                        </div>
                        <div className="absolute left-6 top-1/2 transform -translate-y-1/2 text-yellow-600 text-xl font-black">
                          🔥
                        </div>
                      </div>
                      {parseFloat(amount) <= highestBid && (
                        <div className="flex items-center gap-2 p-3 bg-red-900/80 border-2 border-red-500 rounded-xl">
                          <span className="text-red-400 text-lg">⚠️</span>
                          <p className="text-red-300 text-sm font-bold">
                            Bid must exceed {highestBid.toFixed(4)} ETH
                          </p>
                        </div>
                      )}

                      {/* USD Conversion */}
                      <div className="text-center p-4 bg-yellow-400/10 border border-yellow-400/30 rounded-xl">
                        <span className="text-yellow-300 text-lg font-black">≈ ${(parseFloat(amount || "0") * 2500).toLocaleString()} USD</span>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <label className="block text-lg font-black text-yellow-400 mb-3 flex items-center gap-2">
                        🌍 NETWORK SELECTION
                      </label>
                      <select
                        value={chain}
                        onChange={(e) => setChain(e.target.value as "mainnet" | "sepolia")}
                        className="w-full p-4 border-2 border-yellow-400/60 rounded-2xl focus:border-yellow-400 focus:ring-4 focus:ring-yellow-400/30 transition-all duration-300 bg-gradient-to-r from-yellow-50 to-amber-50 hover:from-yellow-100 hover:to-amber-100 text-lg font-black text-black"
                        disabled={!isActive}
                      >
                        <option value="mainnet">🌐 Ethereum Mainnet</option>
                        <option value="sepolia">🧪 Sepolia Testnet</option>
                      </select>
                    </div>

                    {/* Enhanced Payment Button */}
                    <div className="pt-6">
                      <Payment
                        to={process.env.REACT_APP_RECEIVER_ADDRESS || "0xReceiverAddressHere"}
                        amount={amount}
                        chain={chain}
                        onSuccess={onSuccess}
                        onError={onError}
                        disabled={!isActive || parseFloat(amount) <= highestBid}
                      />
                    </div>
                  </div>

                  {/* Right Column - Yellow-themed Bid History */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-xl font-black text-yellow-400 flex items-center gap-2">
                        📈 LIVE BID FEED
                      </h3>
                      <div className="flex items-center gap-2 px-3 py-1 bg-yellow-400/20 border border-yellow-400/40 rounded-full">
                        <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                        <span className="text-yellow-400 text-xs font-black">LIVE</span>
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-black/60 to-gray-900/80 rounded-2xl p-6 max-h-96 overflow-y-auto border-2 border-yellow-400/30 shadow-inner">
                      {bids.length === 0 ? (
                        <div className="text-center py-12 text-gray-500">
                          <div className="text-5xl mb-4 animate-pulse">🎯</div>
                          <p className="font-bold text-lg mb-2">No bids yet!</p>
                          <p className="text-sm">Be the first to place a bid and start this exciting auction!</p>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          {[...bids].reverse().map((bid, i) => (
                            <div
                              key={i}
                              className={`
                                p-4 rounded-2xl shadow-sm border-2 transition-all duration-300 hover:scale-105
                                ${parseFloat(bid.amount) === highestBid && isActive ?
                                  'bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-400 shadow-yellow-200' :
                                  'bg-white border-gray-200 hover:shadow-md'}
                                ${!isActive && parseFloat(bid.amount) === highestBid ?
                                  'bg-gradient-to-r from-green-50 to-emerald-50 border-green-400 shadow-green-200' : ''}
                              `}
                            >
                              <div className="flex justify-between items-start">
                                <div>
                                  <div className="font-bold text-lg text-gray-800 flex items-center gap-2">
                                    {parseFloat(bid.amount) === highestBid && isActive && (
                                      <span className="text-yellow-500 animate-bounce text-xl">🏆</span>
                                    )}
                                    {!isActive && parseFloat(bid.amount) === highestBid && (
                                      <span className="text-green-500 text-xl">🎉</span>
                                    )}
                                    <span>{bid.amount} ETH</span>
                                  </div>
                                  <div className="text-sm text-gray-600 flex items-center gap-1 mt-1">
                                    {getChainIcon(bid.chain)}
                                    <span className="capitalize">{bid.chain}</span>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <div className="text-sm font-medium text-gray-700 font-mono bg-gray-100 px-2 py-1 rounded">
                                    {shortenAddress(bid.bidder)}
                                  </div>
                                  <div className="text-xs text-gray-500 mt-1">
                                    {new Date(bid.timestamp).toLocaleTimeString()}
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Enhanced Auction Status */}
                {!isActive && (
                  <div className="mt-8 p-8 bg-gradient-to-r from-red-50 to-pink-50 border-2 border-red-200 rounded-3xl text-center shadow-lg">
                    <div className="text-6xl mb-4 animate-bounce">🏁</div>
                    <h3 className="text-2xl font-bold text-red-800 mb-4">Auction Concluded!</h3>
                    {bids.length > 0 ? (
                      <div className="bg-white/80 rounded-2xl p-6 border border-red-200">
                        <p className="text-red-700 text-lg mb-2">
                          🎊 Congratulations to the winner!
                        </p>
                        <div className="font-mono font-bold text-xl text-gray-800 bg-yellow-100 px-4 py-2 rounded-xl inline-block">
                          {shortenAddress(bids[bids.length - 1].bidder)}
                        </div>
                        <p className="text-red-600 mt-2">
                          Winning bid: <span className="font-bold">{highestBid.toFixed(4)} ETH</span>
                        </p>
                      </div>
                    ) : (
                      <p className="text-red-700 text-lg">No bids were placed during this auction</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Premium Features */}
      <div className="mt-16">
        <PremiumFeatures />
      </div>
    </div>
  );
}