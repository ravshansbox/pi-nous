import { readFile } from "node:fs/promises";
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

type NousModel = {
  id: string;
  name: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  contextWindow: number;
  maxTokens: number;
  compat: Record<string, unknown>;
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

type NousModelListResponse = {
  data?: Array<{
    id?: string;
    pricing?: {
      prompt?: number | string;
      completion?: number | string;
      cache_read?: number | string;
      cache_write?: number | string;
      input_cache_read?: number | string;
      input_cache_write?: number | string;
    };
    modalities?: { input?: string[]; output?: string[] };
    architecture?: { modality?: string; input_modalities?: string[]; output_modalities?: string[] };
    context_window?: number;
    context_length?: number;
    max_output_tokens?: number;
    top_provider?: { max_completion_tokens?: number; context_length?: number };
    supported_parameters?: string[];
  }>;
};

type NousCredentials = OAuthCredentials & {
  enterpriseUrl?: string;
  metadata?: {
    refreshToken?: string;
    tokenType?: string;
    scope?: string;
    oauthAccessToken?: string;
    oauthAccessExpiresAt?: number;
    agentKey?: string;
    agentKeyExpiresAt?: string;
    keyId?: string;
    freeTier?: boolean;
    freeTierCheckedAt?: number;
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

function getOAuthAccessToken(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.oauthAccessToken || credentials.access || "";
}

function getAgentKey(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.agentKey || "";
}

function getAgentKeyExpiryMs(credentials: OAuthCredentials): number | undefined {
  const c = credentials as NousCredentials;
  return parseIsoToMs(c.metadata?.agentKeyExpiresAt);
}

function isModelFree(model: { cost?: { input?: number; output?: number } }): boolean {
  return (model.cost?.input || 0) === 0 && (model.cost?.output || 0) === 0;
}

function partitionNousModelsByTier(models: NousModel[], freeTier: boolean): { selectable: NousModel[]; unavailable: NousModel[] } {
  if (!freeTier) return { selectable: models, unavailable: [] };
  const selectable: NousModel[] = [];
  const unavailable: NousModel[] = [];
  for (const model of models) {
    if (isModelFree(model)) selectable.push(model);
    else unavailable.push(model);
  }
  return { selectable, unavailable };
}

function toFiniteNumber(value: unknown): number | undefined {
  const n = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(n) ? n : undefined;
}

function getCachedFreeTier(credentials: OAuthCredentials): boolean | undefined {
  const c = credentials as NousCredentials;
  const checkedAt = c.metadata?.freeTierCheckedAt;
  if (typeof checkedAt !== "number" || !Number.isFinite(checkedAt)) return undefined;
  if (nowMs() - checkedAt > 3 * 60 * 1000) return undefined;
  return c.metadata?.freeTier;
}

async function fetchAccountInfo(accessToken: string): Promise<{ subscription?: { monthly_charge?: number | string; tier?: number | string } }> {
  const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/account`, {
    headers: authHeaders(accessToken),
  });
  if (!response.ok) throw new Error(`Nous account request failed: ${response.status}`);
  return (await response.json()) as { subscription?: { monthly_charge?: number | string; tier?: number | string } };
}

async function loadPersistedNousCredentials(): Promise<NousCredentials | undefined> {
  try {
    const raw = await readFile(`${process.env.HOME}/.pi/agent/auth.json`, "utf8");
    const parsed = JSON.parse(raw) as { nous?: NousCredentials };
    if (!parsed.nous || parsed.nous.type !== "oauth") return undefined;
    return parsed.nous;
  } catch {
    return undefined;
  }
}

async function resolveStartupNousModels(): Promise<NousModel[]> {
  const persisted = await loadPersistedNousCredentials();
  if (!persisted) return fallbackModels();

  const oauthAccessToken = getOAuthAccessToken(persisted);
  const agentKey = getAgentKey(persisted) || persisted.access;
  if (!oauthAccessToken || !agentKey) return fallbackModels();

  const [account, liveModels] = await Promise.all([
    fetchAccountInfo(oauthAccessToken),
    fetchLiveModels(agentKey, safeBaseUrl(persisted.enterpriseUrl)),
  ]);

  const monthlyCharge = toFiniteNumber(account.subscription?.monthly_charge);
  const tier = toFiniteNumber(account.subscription?.tier);
  const freeTier = monthlyCharge === 0 || tier === 5;
  return freeTier ? partitionNousModelsByTier(liveModels, true).selectable : liveModels;
}

async function annotateFreeTier(credentials: NousCredentials): Promise<NousCredentials> {
  const cached = getCachedFreeTier(credentials);
  if (typeof cached === "boolean") return credentials;

  try {
    const account = await fetchAccountInfo(getOAuthAccessToken(credentials));
    const monthlyCharge = toFiniteNumber(account.subscription?.monthly_charge);
    const tier = toFiniteNumber(account.subscription?.tier);
    if (typeof monthlyCharge === "number" || typeof tier === "number") {
      const freeTier = monthlyCharge === 0 || tier === 5;
      return {
        ...credentials,
        metadata: {
          ...credentials.metadata,
          freeTier,
          freeTierCheckedAt: nowMs(),
        },
      };
    }
  } catch {
    return credentials;
  }

  return credentials;
}

function supportsTextInput(live?: NousModelListResponse["data"][number]): boolean {
  const modality = live?.architecture?.modality;
  if (typeof modality === "string" && modality.includes("->")) {
    const [input] = modality.split("->", 1);
    return input.split("+").includes("text");
  }
  const inputs = live?.modalities?.input || live?.architecture?.input_modalities;
  return Array.isArray(inputs) ? inputs.includes("text") : false;
}

function supportsTextOutput(live?: NousModelListResponse["data"][number]): boolean {
  const modality = live?.architecture?.modality;
  if (typeof modality === "string" && modality.includes("->")) {
    const [, output] = modality.split("->", 2);
    return output.split("+").includes("text");
  }
  const outputs = live?.modalities?.output || live?.architecture?.output_modalities;
  return Array.isArray(outputs) ? outputs.includes("text") : false;
}

function supportsToolCalling(live?: NousModelListResponse["data"][number]): boolean {
  return Array.isArray(live?.supported_parameters) && live.supported_parameters.includes("tools");
}

function modelMeta(id: string, live?: NousModelListResponse["data"][number]): NousModel {
  const lower = id.toLowerCase();
  const liveInputs = live?.modalities?.input || live?.architecture?.input_modalities;
  const image = Array.isArray(liveInputs) ? liveInputs.includes("image") : /(claude|gpt|gemini|grok|vision|vl|mimo-v2-omni|glm-5v)/.test(lower);
  const reasoning = /(opus|sonnet|gpt-5|codex|gemini|grok|reason|thinking)/.test(lower);
  let contextWindow = live?.context_window || live?.context_length || live?.top_provider?.context_length || 262144;
  let maxTokens = live?.max_output_tokens || live?.top_provider?.max_completion_tokens || 32768;
  if (!live?.context_window || !live?.max_output_tokens) {
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
  }

  const compat: Record<string, unknown> = { supportsDeveloperRole: false };
  const pricing = live?.pricing;

  return {
    id,
    name: id,
    reasoning,
    input: (image ? ["text", "image"] : ["text"]) as ("text" | "image")[],
    cost: {
      input: toFiniteNumber(pricing?.prompt) || 0,
      output: toFiniteNumber(pricing?.completion) || 0,
      cacheRead: toFiniteNumber(pricing?.cache_read ?? pricing?.input_cache_read) || 0,
      cacheWrite: toFiniteNumber(pricing?.cache_write ?? pricing?.input_cache_write) || 0,
    },
    contextWindow,
    maxTokens,
    compat,
  };
}

async function fetchLiveModels(apiKey: string, baseUrl = NOUS_INFERENCE_URL): Promise<NousModel[]> {
  const response = await fetch(`${safeBaseUrl(baseUrl)}/models`, {
    headers: authHeaders(apiKey),
  });
  if (!response.ok) throw new Error(`Nous models request failed: ${response.status}`);

  const payload = (await response.json()) as NousModelListResponse;
  const models = (payload.data || [])
    .filter((item) => item.id && !item.id.toLowerCase().includes("hermes"))
    .filter((item) => supportsTextInput(item) && supportsTextOutput(item) && supportsToolCalling(item))
    .map((item) => modelMeta(item.id!.trim(), item));

  return models.length ? models : fallbackModels();
}

function fetchTierAwareModels(models: NousModel[], freeTier: boolean): { selectable: NousModel[]; unavailable: NousModel[] } {
  return partitionNousModelsByTier(models, freeTier);
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

  const oauthExpiresAt = nowMs() + (data.expires_in || 3600) * 1000;
  return {
    refresh: data.refresh_token || refreshToken,
    access: data.access_token,
    expires: oauthExpiresAt,
    enterpriseUrl: safeBaseUrl(data.inference_base_url || (credentials as NousCredentials).enterpriseUrl),
    metadata: {
      refreshToken: data.refresh_token || refreshToken,
      tokenType: data.token_type || getTokenType(credentials),
      scope: data.scope || getScope(credentials),
      oauthAccessToken: data.access_token,
      oauthAccessExpiresAt: oauthExpiresAt,
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
        oauthAccessToken: getOAuthAccessToken(baseCreds),
        oauthAccessExpiresAt: baseCreds.expires,
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
      oauthAccessToken: getOAuthAccessToken(baseCreds),
      oauthAccessExpiresAt: baseCreds.expires,
      agentKey: data.api_key,
      agentKeyExpiresAt: data.expires_at,
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

    const oauthExpiresAt = nowMs() + (token.expires_in || 3600) * 1000;
    const credentials: NousCredentials = {
      refresh: token.refresh_token || "",
      access: token.access_token,
      expires: oauthExpiresAt,
      enterpriseUrl: safeBaseUrl(token.inference_base_url),
      metadata: {
        refreshToken: token.refresh_token || "",
        tokenType: token.token_type || "Bearer",
        scope: token.scope || NOUS_SCOPE,
        oauthAccessToken: token.access_token,
        oauthAccessExpiresAt: oauthExpiresAt,
      },
    };

    const minted = await mintAgentKey(credentials);
    const annotated = await annotateFreeTier(minted);
    return {
      ...annotated,
      metadata: {
        ...annotated.metadata,
        oauthAccessToken: token.access_token,
        oauthAccessExpiresAt: oauthExpiresAt,
      },
    };
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
    return await annotateFreeTier(await mintAgentKey(creds));
  }
}

function fallbackModels() {
  return FALLBACK_MODELS.map((id) => modelMeta(id));
}

export default async function (pi: ExtensionAPI) {
  let discoveredModels = fallbackModels();

  try {
    const envApiKey = process.env.NOUS_API_KEY?.trim();
    if (envApiKey) discoveredModels = await fetchLiveModels(envApiKey);
    else discoveredModels = await resolveStartupNousModels();
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
        const minted = await mintAgentKey(refreshed);
        const annotated = await annotateFreeTier(minted);
        return {
          ...annotated,
          metadata: {
            ...annotated.metadata,
            oauthAccessToken: refreshed.access,
            oauthAccessExpiresAt: refreshed.expires,
          },
        };
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
        const c = credentials as NousCredentials;
        const baseUrl = safeBaseUrl(c.enterpriseUrl);
        const freeTier = c.metadata?.freeTier === true;
        const filtered = freeTier ? partitionNousModelsByTier(currentModels as NousModel[], true).selectable : (currentModels as NousModel[]);
        return filtered.map((m) => (m.provider === "nous" ? { ...m, baseUrl } : m));
      },
    },
  });
}
