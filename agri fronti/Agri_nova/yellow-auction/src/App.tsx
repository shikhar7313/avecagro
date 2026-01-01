// src/App.tsx
import React from "react";
import { ConnectButton } from "@rainbow-me/rainbowkit";
import Auction from "./components/Auction";
import { useNotification } from "./components/NotificationToast";

export default function App() {
  const { NotificationContainer, showNotification } = useNotification();

  return (
    <div className="min-h-screen relative overflow-hidden bg-brand-dark text-white">
      {/* Background Layered Gradients & Grid */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,#facc1533,transparent_60%)]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_70%,#f59e0b22,transparent_65%)]"></div>
        <div className="absolute inset-0 bg-grid-yellow opacity-10 mix-blend-overlay"></div>
        {/* Subtle animated accent rings */}
        <div className="pointer-events-none absolute -top-24 -left-24 w-96 h-96 rounded-full border border-brand-yellow-400/20 animate-pulse"></div>
        <div className="pointer-events-none absolute bottom-0 -right-40 w-[38rem] h-[38rem] rounded-full border border-brand-yellow-400/10 animate-spin-slow"></div>
      </div>

      {/* Header */}
      <div className="relative z-20 bg-[#0f0f0f]/90 backdrop-blur-md border-b border-brand-yellow-400/40 shadow-[0_1px_0_0_rgba(250,204,21,0.25)]">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center bg-gradient-to-br from-brand-yellow-400 to-brand-amber-500 shadow-brand-glow font-black text-xl text-brand-dark">Y</div>
              <div className="leading-tight">
                <h1 className="text-[1.55rem] font-extrabold tracking-tight text-gradient-brand">YELLOW AUCTION</h1>
                <p className="text-xs md:text-sm font-medium text-brand-yellow-200/80">Premium Digital Assets Marketplace</p>
              </div>
            </div>

            {/* Wallet Button Wrapper */}
            <div className="rounded-xl p-[2px] bg-gradient-to-r from-brand-yellow-400 via-brand-amber-500 to-brand-yellow-400 shadow-brand-glow hover:scale-[1.02] transition-transform">
              <div className="bg-[#111]/95 rounded-lg px-2 py-1">
                <ConnectButton />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
          {/* Hero Section */}
          <div className="text-center mb-16 animate-fade-in">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-3xl sm:text-5xl lg:text-6xl font-black mb-6 leading-tight tracking-tight">
                <span className="text-gradient-brand">Bid. Win. Own.</span>
              </h2>
              <p className="text-lg sm:text-xl text-brand-yellow-100/90 font-medium mb-10 leading-relaxed max-w-2xl mx-auto">
                A curated auction experience for exclusive digital assets powered by secure cross‑chain settlement.
              </p>

              {/* Feature Pills */}
              <div className="flex flex-wrap justify-center gap-4 mb-8">
                {[
                  ['🔒 Secure Bidding', 'Audited smart contract flows'],
                  ['⚡ Instant Settlement', 'Low-latency confirmations'],
                  ['🌐 Cross-Chain Ready', 'Multi-network support']
                ].map(([title, subtitle]) => (
                  <div key={title} className="group relative overflow-hidden rounded-2xl border border-brand-yellow-400/30 bg-[#141414]/70 px-5 py-4 backdrop-blur-sm hover:border-brand-yellow-400/60 transition">
                    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition bg-gradient-to-br from-brand-yellow-400/10 to-transparent" />
                    <div className="relative">
                      <div className="font-semibold text-sm text-brand-yellow-200 flex items-center gap-2">{title}</div>
                      <div className="text-[11px] mt-1 tracking-wide text-brand-yellow-100/60 font-medium">{subtitle}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Auction Component - Centered */}
          <main className="flex justify-center">
            <div className="w-full max-w-7xl">
              <Auction showNotification={showNotification} />
            </div>
          </main>

          {/* Footer */}
          <footer className="mt-20 text-center text-[12px] text-brand-yellow-100/60">
            <div className="flex flex-wrap items-center justify-center gap-5 mb-5">
              {[
                ['Secure', 'green'],
                ['Fast', 'amber'],
                ['Cross-Chain', 'cyan']
              ].map(([label, color]) => (
                <div key={label} className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full animate-pulse bg-${color}-400`}></span>
                  <span className="tracking-wide uppercase text-[11px] font-semibold">{label}</span>
                </div>
              ))}
            </div>
            <p className="tracking-wide">© 2025 Yellow Auction • All Rights Reserved</p>
          </footer>
        </div>
      </div>

      {/* Notification Container */}
      <NotificationContainer />
    </div>
  );
}
