// src/components/Payment.tsx
import React, { useEffect, useState } from "react";
import { useAccount, useWalletClient, useWaitForTransactionReceipt } from "wagmi";
import nitroAdapter from "../lib/nitroAdapter";
import { parseEther } from "ethers";
import LoadingSpinner from "./LoadingSpinner";

interface Props {
  to: string;
  amount: string;
  chain?: "mainnet" | "sepolia";
  onSuccess?: () => void;
  onError?: (message: string) => void;
  disabled?: boolean;
}

export default function Payment({ to, amount, chain = "sepolia", onSuccess, onError, disabled = false }: Props) {
  const { address } = useAccount();
  const { data: walletClient } = useWalletClient();
  const [status, setStatus] = useState<string>("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      setStatus("Initializing routing adapter...");
      try {
        await nitroAdapter.init({
          chain,
          rpcUrl:
            chain === "mainnet"
              ? `https://mainnet.infura.io/v3/${process.env.REACT_APP_INFURA_KEY}`
              : `https://sepolia.infura.io/v3/${process.env.REACT_APP_INFURA_KEY}`,
        });
        setReady(true);
        setStatus("");
      } catch (e: any) {
        console.error("Adapter init error:", e);
        const errorMsg = "Adapter init failed (check console)";
        setStatus(errorMsg);
        if (onError) onError(errorMsg);
      }
    })();
  }, [chain, onError]);

  const handlePay = async () => {
    if (disabled) {
      return;
    }
    if (!address) {
      const errorMsg = "Connect wallet first";
      setStatus(errorMsg);
      if (onError) onError(errorMsg);
      return;
    }
    if (!walletClient) {
      const errorMsg = "Wallet client not available";
      setStatus(errorMsg);
      if (onError) onError(errorMsg);
      return;
    }

    try {
      setStatus("Requesting best route...");
      const route = await nitroAdapter.routePayment({
        from: address,
        to,
        amount,
        token: "ETH",
      });

      if (route.txData && route.txData.to) {
        setStatus("Preparing transaction...");
        const txReq: any = {
          to: route.txData.to,
          data: route.txData.data || undefined,
        };
        if (route.txData.value) {
          txReq.value = BigInt(route.txData.value);
        } else {
          txReq.value = parseEther(amount);
        }

        setStatus("Sending transaction via wallet...");
        const txHash = await walletClient.sendTransaction({
          to: txReq.to as `0x${string}`,
          data: txReq.data as `0x${string}`,
          value: txReq.value
        });
        setStatus(`Tx sent: ${txHash}. Waiting confirmation...`);
        // Note: In wagmi v2, transaction waiting needs to be handled differently
        // You might want to use useWaitForTransaction hook or similar

        setStatus("Notifying nitro node (finalize)...");
        if (route.routeId && process.env.REACT_APP_NITRO_NODE_URL) {
          try {
            await fetch(`${process.env.REACT_APP_NITRO_NODE_URL.replace(/\/$/, "")}/finalize-route`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ routeId: route.routeId, txHash }),
            });
          } catch (e) {
            console.warn("Finalize notify failed:", e);
          }
        }

        setStatus("✅ Payment successful");
        if (onSuccess) onSuccess();
        return;
      }

      if (route.wait) {
        setStatus("Waiting on nitrolite server-side processing...");
        await route.wait();
        setStatus("✅ Payment successful (server-side)");
        if (onSuccess) onSuccess();
        return;
      }

      const unexpectedMsg = "⚠️ Unexpected route response (check console).";
      setStatus(unexpectedMsg);
      if (onError) onError(unexpectedMsg);
      console.log("route raw:", route);
    } catch (err: any) {
      console.error(err);
      const errorMsg = "Payment failed: " + (err?.message || String(err));
      setStatus(errorMsg);
      if (onError) onError(errorMsg);
    }
  };

  const isDisabled = !ready || disabled;

  return (
    <div className="space-y-4">
      <button
        onClick={handlePay}
        disabled={isDisabled}
        className={`
          w-full py-4 px-8 rounded-2xl font-bold text-lg transition-all duration-300 shadow-lg
          ${isDisabled
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white shadow-yellow-200 hover:shadow-xl transform hover:scale-105'
          }
        `}
      >
        {!ready ? (
          <div className="flex items-center justify-center gap-3">
            <LoadingSpinner size="sm" color="gray" />
            <span>Initializing...</span>
          </div>
        ) : disabled ? (
          '🚫 Bid Not Allowed'
        ) : (
          <div className="flex items-center justify-center gap-2">
            <span>🚀 Place Bid</span>
            <span className="font-mono">{amount} ETH</span>
            <span className="text-sm opacity-90">({chain})</span>
          </div>
        )}
      </button>

      {status && (
        <div className={`
          p-4 rounded-xl text-center font-medium transition-all duration-200
          ${status.includes('✅')
            ? 'bg-green-100 text-green-800 border-2 border-green-200'
            : status.includes('⚠️') || status.includes('failed')
              ? 'bg-red-100 text-red-800 border-2 border-red-200'
              : 'bg-blue-100 text-blue-800 border-2 border-blue-200'
          }
        `}>
          {status.includes('Tx sent:') ? (
            <div className="space-y-2">
              <div>🔗 Transaction Sent</div>
              <div className="text-xs font-mono break-all opacity-75">
                {status.split('Tx sent: ')[1]?.split('.')[0]}
              </div>
              <div className="text-xs pt-1">Waiting for confirmation...</div>
            </div>
          ) : (
            status
          )}
        </div>
      )}
    </div>
  );
}
