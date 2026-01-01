// src/lib/nitroAdapter.ts
// Adapter: uses @erc7824/nitrolite package if available, otherwise HTTP node (REACT_APP_NITRO_NODE_URL).

type InitConfig = {
  chain: string;
  rpcUrl: string;
  apiKey?: string;
};

type RouteParams = {
  from: string;
  to: string;
  amount: string;
  token?: string;
};

type RouteResult = {
  txData?: { to: string; data?: string; value?: string };
  routeId?: string;
  wait?: () => Promise<any>;
  raw?: any;
};

let nitroPkg: any = null;
let cfgUsed: InitConfig | null = null;
const nodeUrl = process.env.REACT_APP_NITRO_NODE_URL || "";

async function tryImport() {
  if (nitroPkg !== null) return nitroPkg;
  try {
    // dynamic import
    // @ts-ignore
    nitroPkg = await import("@erc7824/nitrolite");
    return nitroPkg;
  } catch (e) {
    nitroPkg = null;
    return null;
  }
}

export async function init(cfg: InitConfig) {
  cfgUsed = cfg;
  const pkg = await tryImport();
  if (pkg && typeof pkg.init === "function") {
    try {
      await pkg.init({ rpc: cfg.rpcUrl, chain: cfg.chain, apiKey: cfg.apiKey });
      return { mode: "package" };
    } catch (err) {
      console.warn("nitro package init failed:", err);
    }
  }
  // fallback: just return http mode
  return { mode: "http", nodeUrl };
}

async function callNodeRoute(params: RouteParams): Promise<RouteResult> {
  if (!nodeUrl) throw new Error("No REACT_APP_NITRO_NODE_URL configured for HTTP fallback.");
  const url = `${nodeUrl.replace(/\/$/, "")}/route-payment`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...params, chain: cfgUsed?.chain || "sepolia" }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Nitro node error: ${res.status} ${text}`);
  }
  const json = await res.json();
  return { txData: json.txData, routeId: json.routeId, raw: json };
}

export async function routePayment(params: RouteParams): Promise<RouteResult> {
  const pkg = await tryImport();
  if (pkg && typeof pkg.routePayment === "function") {
    try {
      const r = await pkg.routePayment(params);
      return r as RouteResult;
    } catch (err) {
      console.warn("nitro package routePayment failed:", err);
    }
  }
  return await callNodeRoute(params);
}

export default {
  init,
  routePayment,
};
