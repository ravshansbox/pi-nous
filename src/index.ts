import type { OAuthCredentials, OAuthLoginCallbacks } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

const NOUS_PORTAL_URL = "https://portal.nousresearch.com";
const NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1";
const NOUS_CLIENT_ID = "hermes-cli";
const NOUS_SCOPE = "inference:mint_agent_key";
const ACCESS_REFRESH_SKEW_MS = 2 * 60 * 1000;
const AGENT_KEY_MIN_TTL_SECONDS = 30 * 60;
const DEVICE_POLL_INTERVAL_CAP_MS = 1000;
const BROWSER_LIKE_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36";
const FALLBACK_MODELS = [
  "moonshotai/kimi-k2.6",
  "xiaomi/mimo-v2.5-pro",
  "xiaomi/mimo-v2.5",
  "anthropic/claude-opus-4.7",
  "anthropic/claude-opus-4.6",
  "anthropic/claude-sonnet-4.6",
  "anthropic/claude-sonnet-4.5",
  "anthropic/claude-haiku-4.5",
  "openai/gpt-5.4",
  "openai/gpt-5.4-mini",
  "openai/gpt-5.3-codex",
  "google/gemini-3-pro-preview",
  "google/gemini-3-flash-preview",
  "google/gemini-3.1-pro-preview",
  "google/gemini-3.1-flash-lite-preview",
  "qwen/qwen3.5-plus-02-15",
  "qwen/qwen3.5-35b-a3b",
  "stepfun/step-3.5-flash",
  "minimax/minimax-m2.7",
  "minimax/minimax-m2.5",
  "z-ai/glm-5.1",
  "z-ai/glm-5v-turbo",
  "z-ai/glm-5-turbo",
  "x-ai/grok-4.20-beta",
  "nvidia/nemotron-3-super-120b-a12b",
  "arcee-ai/trinity-large-thinking",
  "openai/gpt-5.4-pro",
  "openai/gpt-5.4-nano",
] as const;

type NousTokenResponse = {
  access_token: string;
  refresh_token?: string;
  token_type?: string;
  expires_in?: number;
  scope?: string;
  inference_base_url?: string;
  error?: string;
  error_description?: string;
};

type NousDeviceCodeResponse = {
  device_code: string;
  user_code: string;
  verification_uri: string;
  verification_uri_complete?: string;
  expires_in: number;
  interval?: number;
};

type NousAgentKeyResponse = {
  api_key: string;
  key_id?: string;
  expires_at?: string;
  expires_in?: number;
  reused?: boolean;
  inference_base_url?: string;
};

type NousCredentials = OAuthCredentials & {
  enterpriseUrl?: string;
  metadata?: {
    refreshToken?: string;
    tokenType?: string;
    scope?: string;
    agentKey?: string;
    agentKeyExpiresAt?: string;
    agentKeyExpiresIn?: number;
    keyId?: string;
  };
};

function nowMs() {
  return Date.now();
}

function authHeaders(token?: string): Record<string, string> {
  const base: Record<string, string> = {
    Accept: "application/json",
    "User-Agent": BROWSER_LIKE_UA,
  };
  if (token) base.Authorization = `Bearer ${token}`;
  return base;
}

function safeBaseUrl(url?: string): string {
  return (url || NOUS_INFERENCE_URL).replace(/\/+$/, "");
}

function isExpiring(expiresAtMs?: number, skewMs = 0): boolean {
  if (!expiresAtMs || !Number.isFinite(expiresAtMs)) return true;
  return expiresAtMs - skewMs <= nowMs();
}

function parseIsoToMs(value?: string): number | undefined {
  if (!value) return undefined;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : undefined;
}

function getRefreshToken(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.refreshToken || credentials.refresh || "";
}

function getTokenType(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.tokenType || "Bearer";
}

function getScope(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.scope || NOUS_SCOPE;
}

function getAgentKey(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.agentKey || "";
}

function getAgentKeyExpiryMs(credentials: OAuthCredentials): number | undefined {
  const c = credentials as NousCredentials;
  const meta = c.metadata;
  if (!meta) return undefined;
  if (typeof meta.agentKeyExpiresIn === "number" && Number.isFinite(meta.agentKeyExpiresIn)) {
    return nowMs() + meta.agentKeyExpiresIn * 1000;
  }
  return parseIsoToMs(meta.agentKeyExpiresAt);
}

function modelMeta(id: string) {
  const lower = id.toLowerCase();
  const image = /(claude|gpt|gemini|grok|vision|vl|mimo-v2-omni|glm-5v)/.test(lower);
  const reasoning = /(opus|sonnet|gpt-5|codex|gemini|grok|reason|thinking)/.test(lower);
  let contextWindow = 262144;
  let maxTokens = 32768;
  if (lower.includes("gemini")) {
    contextWindow = 1048576;
    maxTokens = 65536;
  } else if (lower.includes("gpt-5")) {
    contextWindow = 400000;
  } else if (lower.includes("haiku")) {
    maxTokens = 8192;
    contextWindow = 200000;
  } else if (lower.includes("sonnet") || lower.includes("opus")) {
    contextWindow = 200000;
    maxTokens = lower.includes("opus") ? 32000 : 16384;
  }

  const compat: Record<string, unknown> = { supportsDeveloperRole: false };
  if (lower.includes("gemini")) compat.supportsDeveloperRole = false;

  return {
    id,
    name: id,
    reasoning,
    input: (image ? ["text", "image"] : ["text"]) as ("text" | "image")[],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens,
    compat,
  };
}

async function requestDeviceCode(): Promise<NousDeviceCodeResponse> {
  const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/device/code`, {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/x-www-form-urlencoded",
      Origin: NOUS_PORTAL_URL,
      Referer: `${NOUS_PORTAL_URL}/`,
    },
    body: new URLSearchParams({
      client_id: NOUS_CLIENT_ID,
      scope: NOUS_SCOPE,
    }).toString(),
  });

  const text = await response.text();
  if (!response.ok) {
    const lower = text.toLowerCase();
    if (response.status === 429 && (lower.includes("vercel security checkpoint") || lower.includes("verifying your browser"))) {
      throw new Error("NOUS_VERCEL_CHECKPOINT");
    }
    throw new Error(`Nous device code request failed: ${response.status} ${text}`);
  }

  const data = JSON.parse(text) as NousDeviceCodeResponse;
  if (!data.device_code || !data.user_code || !data.verification_uri) {
    throw new Error("Nous device code response missing required fields");
  }
  return data;
}

async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error("Login cancelled"));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new Error("Login cancelled"));
      },
      { once: true },
    );
  });
}

async function pollForToken(device: NousDeviceCodeResponse, signal?: AbortSignal): Promise<NousTokenResponse> {
  const deadline = nowMs() + device.expires_in * 1000;
  let intervalMs = Math.max(1000, Math.min((device.interval || 5) * 1000, DEVICE_POLL_INTERVAL_CAP_MS));

  while (nowMs() < deadline) {
    if (signal?.aborted) throw new Error("Login cancelled");

    const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/token`, {
      method: "POST",
      headers: {
        ...authHeaders(),
        "Content-Type": "application/x-www-form-urlencoded",
        Origin: NOUS_PORTAL_URL,
        Referer: `${NOUS_PORTAL_URL}/`,
      },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
        client_id: NOUS_CLIENT_ID,
        device_code: device.device_code,
      }).toString(),
    });

    const text = await response.text();
    let data: NousTokenResponse | null = null;
    try {
      data = text ? (JSON.parse(text) as NousTokenResponse) : null;
    } catch {
      data = null;
    }

    if (response.ok && data?.access_token) return data;

    const error = data?.error;
    if (error === "authorization_pending") {
      await sleep(intervalMs, signal);
      continue;
    }
    if (error === "slow_down") {
      intervalMs = Math.min(intervalMs + 5000, 10000);
      await sleep(intervalMs, signal);
      continue;
    }
    if (error === "expired_token") {
      throw new Error("Nous device code expired. Please try /login again.");
    }
    if (error === "access_denied") {
      throw new Error("Nous authorization was denied.");
    }

    throw new Error(`Nous token request failed: ${response.status} ${text}`);
  }

  throw new Error("Nous login timed out.");
}

async function refreshAccessToken(credentials: OAuthCredentials): Promise<NousCredentials> {
  const refreshToken = getRefreshToken(credentials);
  if (!refreshToken) throw new Error("No Nous refresh token available");

  const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/token`, {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/x-www-form-urlencoded",
      Origin: NOUS_PORTAL_URL,
      Referer: `${NOUS_PORTAL_URL}/`,
    },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      client_id: NOUS_CLIENT_ID,
      refresh_token: refreshToken,
    }).toString(),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Nous token refresh failed: ${response.status} ${text}`);
  }

  const data = JSON.parse(text) as NousTokenResponse;
  if (!data.access_token) {
    throw new Error("Nous token refresh succeeded without access_token");
  }

  return {
    refresh: data.refresh_token || refreshToken,
    access: data.access_token,
    expires: nowMs() + (data.expires_in || 3600) * 1000,
    enterpriseUrl: safeBaseUrl(data.inference_base_url || (credentials as NousCredentials).enterpriseUrl),
    metadata: {
      refreshToken: data.refresh_token || refreshToken,
      tokenType: data.token_type || getTokenType(credentials),
      scope: data.scope || getScope(credentials),
    },
  };
}

async function mintAgentKey(credentials: OAuthCredentials): Promise<NousCredentials> {
  let baseCreds = credentials as NousCredentials;
  if (isExpiring(credentials.expires, ACCESS_REFRESH_SKEW_MS)) {
    baseCreds = await refreshAccessToken(credentials);
  }

  const existingKey = getAgentKey(baseCreds);
  const existingKeyExpiry = getAgentKeyExpiryMs(baseCreds);
  if (existingKey && existingKeyExpiry && !isExpiring(existingKeyExpiry, 60 * 1000)) {
    return {
      ...baseCreds,
      access: existingKey,
      enterpriseUrl: safeBaseUrl(baseCreds.enterpriseUrl),
      metadata: {
        ...baseCreds.metadata,
        refreshToken: getRefreshToken(baseCreds),
        tokenType: getTokenType(baseCreds),
        scope: getScope(baseCreds),
        agentKey: existingKey,
      },
    };
  }

  const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/agent-key`, {
    method: "POST",
    headers: {
      ...authHeaders(baseCreds.access),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ min_ttl_seconds: AGENT_KEY_MIN_TTL_SECONDS }),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Nous agent key mint failed: ${response.status} ${text}`);
  }

  const data = JSON.parse(text) as NousAgentKeyResponse;
  if (!data.api_key) {
    throw new Error("Nous agent key response missing api_key");
  }

  return {
    ...baseCreds,
    access: data.api_key,
    enterpriseUrl: safeBaseUrl(data.inference_base_url || baseCreds.enterpriseUrl),
    metadata: {
      ...baseCreds.metadata,
      refreshToken: getRefreshToken(baseCreds),
      tokenType: getTokenType(baseCreds),
      scope: getScope(baseCreds),
      agentKey: data.api_key,
      agentKeyExpiresAt: data.expires_at,
      agentKeyExpiresIn: data.expires_in,
      keyId: data.key_id,
    },
  };
}

async function loginNous(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
  try {
    const device = await requestDeviceCode();

    callbacks.onAuth({ url: device.verification_uri_complete || device.verification_uri });
    await callbacks.onPrompt({
      message: `Open the URL in your browser and approve access. If prompted, enter this code: ${device.user_code}. Press Enter here after approving to continue polling.`,
    });

    const token = await pollForToken(device, callbacks.signal);
    if (!token.access_token) throw new Error("Nous login did not return an access token");

    const credentials: NousCredentials = {
      refresh: token.refresh_token || "",
      access: token.access_token,
      expires: nowMs() + (token.expires_in || 3600) * 1000,
      enterpriseUrl: safeBaseUrl(token.inference_base_url),
      metadata: {
        refreshToken: token.refresh_token || "",
        tokenType: token.token_type || "Bearer",
        scope: token.scope || NOUS_SCOPE,
      },
    };

    return await mintAgentKey(credentials);
  } catch (error) {
    if (!(error instanceof Error) || error.message !== "NOUS_VERCEL_CHECKPOINT") throw error;

    callbacks.onAuth({ url: `${NOUS_PORTAL_URL}/login` });
    const pasted = await callbacks.onPrompt({
      message:
        "Nous Portal is behind a Vercel browser checkpoint. Open the login page in your normal browser, complete login there, then paste a Nous inference agent key or access token:",
    });

    const token = pasted.trim();
    if (!token) throw new Error("No token provided");

    if (token.startsWith("sk-")) {
      return {
        refresh: "",
        access: token,
        expires: nowMs() + 24 * 60 * 60 * 1000,
        enterpriseUrl: safeBaseUrl(NOUS_INFERENCE_URL),
        metadata: {
          tokenType: "Bearer",
          scope: NOUS_SCOPE,
          agentKey: token,
        },
      };
    }

    const creds: NousCredentials = {
      refresh: "",
      access: token,
      expires: nowMs() + 60 * 60 * 1000,
      enterpriseUrl: safeBaseUrl(NOUS_INFERENCE_URL),
      metadata: {
        tokenType: "Bearer",
        scope: NOUS_SCOPE,
      },
    };
    return await mintAgentKey(creds);
  }
}

function fallbackModels() {
  return FALLBACK_MODELS.map((id) => modelMeta(id));
}

export default async function (pi: ExtensionAPI) {
  let discoveredModels = fallbackModels();

  try {
    const envApiKey = process.env.NOUS_API_KEY?.trim();
    if (envApiKey) {
      const response = await fetch(`${NOUS_INFERENCE_URL}/models`, {
        headers: authHeaders(envApiKey),
      });
      if (response.ok) {
        const payload = (await response.json()) as { data?: Array<{ id?: string }> };
        const ids = (payload.data || [])
          .map((item) => item?.id?.trim())
          .filter((id): id is string => Boolean(id && !id.toLowerCase().includes("hermes")));
        if (ids.length) discoveredModels = [...new Set(ids)].map((id) => modelMeta(id));
      }
    }
  } catch {
    // Keep fallback models. Runtime OAuth flow is the primary path.
  }

  pi.registerProvider("nous", {
    baseUrl: NOUS_INFERENCE_URL,
    api: "openai-completions",
    models: discoveredModels,
    oauth: {
      name: "Nous Portal",
      login: loginNous,
      refreshToken: async (credentials) => {
        const refreshed = await refreshAccessToken(credentials);
        return mintAgentKey(refreshed);
      },
      getApiKey: (credentials) => {
        const c = credentials as NousCredentials;
        return c.metadata?.agentKey || credentials.access;
      },
      modifyModels: (models, credentials) => {
        const currentModels = Array.isArray(models)
          ? models
          : Array.isArray((models as { models?: unknown })?.models)
            ? ((models as { models: typeof discoveredModels }).models)
            : discoveredModels;
        const baseUrl = safeBaseUrl((credentials as NousCredentials).enterpriseUrl);
        return currentModels.map((m) => (m.provider === "nous" ? { ...m, baseUrl } : m));
      },
    },
  });
}
