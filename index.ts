import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { Api, Model, OAuthCredentials, OAuthLoginCallbacks } from '@earendil-works/pi-ai';
import { getAgentDir, type ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';
import { stream as openaiCompletionsStream } from '@earendil-works/pi-ai/api/openai-completions';
import type {
  AssistantMessage,
  AssistantMessageEventStream,
  SimpleStreamOptions,
  TranscriptContext,
} from '@earendil-works/pi-ai';

const NOUS_PORTAL_URL = 'https://portal.nousresearch.com';
const NOUS_INFERENCE_URL = 'https://inference-api.nousresearch.com/v1';
let autoSwitchEnabled: boolean | undefined;
const NOUS_CLIENT_ID = 'hermes-cli';
const NOUS_SCOPE = 'inference:mint_agent_key';
const ACCESS_REFRESH_SKEW_MS = 2 * 60 * 1000;
const DEVICE_POLL_INTERVAL_CAP_MS = 1000;
const BROWSER_LIKE_UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36';
const FALLBACK_MODELS = [
  'moonshotai/kimi-k2.6',
  'xiaomi/mimo-v2.5-pro',
  'xiaomi/mimo-v2.5',
  'anthropic/claude-opus-4.7',
  'anthropic/claude-opus-4.6',
  'anthropic/claude-sonnet-4.6',
  'anthropic/claude-sonnet-4.5',
  'anthropic/claude-haiku-4.5',
  'openai/gpt-5.4',
  'openai/gpt-5.4-mini',
  'openai/gpt-5.3-codex',
  'google/gemini-3-pro-preview',
  'google/gemini-3-flash-preview',
  'google/gemini-3.1-pro-preview',
  'google/gemini-3.1-flash-lite-preview',
  'qwen/qwen3.5-plus-02-15',
  'qwen/qwen3.5-35b-a3b',
  'stepfun/step-3.5-flash',
  'minimax/minimax-m2.7',
  'minimax/minimax-m2.5',
  'z-ai/glm-5.1',
  'z-ai/glm-5v-turbo',
  'z-ai/glm-5-turbo',
  'x-ai/grok-4.20-beta',
  'nvidia/nemotron-3-super-120b-a12b',
  'arcee-ai/trinity-large-thinking',
  'openai/gpt-5.4-pro',
  'openai/gpt-5.4-nano',
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
  input: ('text' | 'image')[];
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
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

type NousModelListItem = {
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
  architecture?: {
    modality?: string;
    input_modalities?: string[];
    output_modalities?: string[];
  };
  context_window?: number;
  context_length?: number;
  max_output_tokens?: number;
  top_provider?: { max_completion_tokens?: number; context_length?: number };
  supported_parameters?: string[];
};

type NousModelListResponse = { data?: NousModelListItem[] };

type NousModelConfig = NousModel & Model<Api> & { provider?: string; baseUrl?: string };

type NousCredentials = OAuthCredentials & {
  enterpriseUrl?: string;
  metadata?: {
    refreshToken?: string | undefined;
    tokenType?: string | undefined;
    scope?: string | undefined;
    oauthAccessToken?: string | undefined;
    oauthAccessExpiresAt?: number | undefined;
    agentKey?: string | undefined;
    agentKeyExpiresAt?: string | undefined;
    keyId?: string | undefined;
    freeTier?: boolean | undefined;
    freeTierCheckedAt?: number | undefined;
  };
};

function nowMs() {
  return Date.now();
}

function authHeaders(token?: string): Record<string, string> {
  const base: Record<string, string> = {
    Accept: 'application/json',
    'User-Agent': BROWSER_LIKE_UA,
  };
  if (token) base['Authorization'] = `Bearer ${token}`;
  return base;
}

function safeBaseUrl(url?: string): string {
  return (url || NOUS_INFERENCE_URL).replace(/\/+$/, '');
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
  return c.metadata?.refreshToken || credentials.refresh || '';
}

function getTokenType(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.tokenType || 'Bearer';
}

function getScope(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.scope || NOUS_SCOPE;
}

function getOAuthAccessToken(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.oauthAccessToken || credentials.access || '';
}

function getAgentKey(credentials: OAuthCredentials): string {
  const c = credentials as NousCredentials;
  return c.metadata?.agentKey || '';
}

function getAgentKeyExpiryMs(credentials: OAuthCredentials): number | undefined {
  const c = credentials as NousCredentials;
  return parseIsoToMs(c.metadata?.agentKeyExpiresAt);
}

// ── JWT minting helpers (per nous_jwt_mining.md) ──
// The Portal's access_token IS the invoke JWT — no separate agent-key endpoint.
// Validate: exp claim + scope claim + minimum TTL before accepting as usable.

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  try {
    const base64 = parts[1] ?? '';
    const json = Buffer.from(base64.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString(
      'utf8',
    );
    return JSON.parse(json) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function getJwtExpMs(token: string): number | undefined {
  const payload = decodeJwtPayload(token);
  if (!payload?.['exp']) return undefined;
  const exp = payload['exp'] as number;
  return exp * 1000;
}

function isModelFree(model: { cost?: { input?: number; output?: number } }): boolean {
  return (model.cost?.input || 0) === 0 && (model.cost?.output || 0) === 0;
}

function partitionNousModelsByTier(
  models: NousModel[],
  freeTier: boolean,
): { selectable: NousModel[]; unavailable: NousModel[] } {
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
  const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN;
  return Number.isFinite(n) ? n : undefined;
}

function getCachedFreeTier(credentials: OAuthCredentials): boolean | undefined {
  const c = credentials as NousCredentials;
  const checkedAt = c.metadata?.freeTierCheckedAt;
  if (typeof checkedAt !== 'number' || !Number.isFinite(checkedAt)) return undefined;
  if (nowMs() - checkedAt > 3 * 60 * 1000) return undefined;
  return c.metadata?.freeTier;
}

async function fetchAccountInfo(accessToken: string): Promise<{
  subscription?: { monthly_charge?: number | string; tier?: number | string };
}> {
  const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/account`, {
    headers: authHeaders(accessToken),
  });
  if (!response.ok) throw new Error(`Nous account request failed: ${response.status}`);
  return (await response.json()) as {
    subscription?: { monthly_charge?: number | string; tier?: number | string };
  };
}

async function loadPersistedNousCredentials(): Promise<NousCredentials | undefined> {
  try {
    const raw = await readFile(join(getAgentDir(), 'auth.json'), 'utf8');
    const parsed = JSON.parse(raw) as { nous?: NousCredentials };
    if (!parsed.nous || parsed.nous['type'] !== 'oauth') return undefined;
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
  if (typeof cached === 'boolean') return credentials;

  try {
    const account = await fetchAccountInfo(getOAuthAccessToken(credentials));
    const monthlyCharge = toFiniteNumber(account.subscription?.monthly_charge);
    const tier = toFiniteNumber(account.subscription?.tier);
    if (typeof monthlyCharge === 'number' || typeof tier === 'number') {
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

function supportsTextInput(live?: NousModelListItem): boolean {
  const modality = live?.architecture?.modality;
  if (typeof modality === 'string' && modality.includes('->')) {
    const [input] = modality.split('->', 1);
    return (input ?? '').split('+').includes('text');
  }
  const inputs = live?.modalities?.input || live?.architecture?.input_modalities;
  if (Array.isArray(inputs)) return inputs.includes('text');
  const id = live?.id?.toLowerCase() || '';
  return !/(image-only|vision-only|audio-only)/.test(id);
}

function supportsTextOutput(live?: NousModelListItem): boolean {
  const modality = live?.architecture?.modality;
  if (typeof modality === 'string' && modality.includes('->')) {
    const [, output] = modality.split('->', 2);
    return (output ?? '').split('+').includes('text');
  }
  const outputs = live?.modalities?.output || live?.architecture?.output_modalities;
  if (Array.isArray(outputs)) return outputs.includes('text');
  return true;
}

function supportsToolCalling(live?: NousModelListItem): boolean {
  if (Array.isArray(live?.supported_parameters) && live.supported_parameters.includes('tools')) {
    return true;
  }
  const id = live?.id?.toLowerCase() || '';
  return /^(openai\/gpt-|anthropic\/claude|google\/gemini|x-ai\/grok|qwen|stepfun|minimax|z-ai\/|xiaomi\/mimo|nvidia\/|moonshotai|arcee-ai)/.test(
    id,
  );
}

function modelMeta(id: string, live?: NousModelListItem): NousModel {
  const lower = id.toLowerCase();
  const liveInputs = live?.modalities?.input || live?.architecture?.input_modalities;
  const image = Array.isArray(liveInputs)
    ? liveInputs.includes('image')
    : /(claude|gpt|gemini|grok|vision|vl|mimo-v2-omni|glm-5v)/.test(lower);
  const reasoning = /(opus|sonnet|gpt-5|codex|gemini|grok|reason|thinking)/.test(lower);
  let contextWindow =
    live?.context_window || live?.context_length || live?.top_provider?.context_length || 262144;
  let maxTokens = live?.max_output_tokens || live?.top_provider?.max_completion_tokens || 32768;
  if (!live?.context_window || !live?.max_output_tokens) {
    if (lower.includes('gemini')) {
      contextWindow = 1048576;
      maxTokens = 65536;
    } else if (lower.includes('gpt-5')) {
      contextWindow = 400000;
    } else if (lower.includes('haiku')) {
      maxTokens = 8192;
      contextWindow = 200000;
    } else if (lower.includes('sonnet') || lower.includes('opus')) {
      contextWindow = 200000;
      maxTokens = lower.includes('opus') ? 32000 : 16384;
    }
  }

  const compat: Record<string, unknown> = { supportsDeveloperRole: false };
  const pricing = live?.pricing;

  return {
    id,
    name: id,
    reasoning,
    input: (image ? ['text', 'image'] : ['text']) as ('text' | 'image')[],
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
    .filter((item) => item.id && !item.id.toLowerCase().includes('hermes'))
    .filter(
      (item) => supportsTextInput(item) && supportsTextOutput(item) && supportsToolCalling(item),
    )
    .map((item) => modelMeta(item.id!.trim(), item));

  return models.length ? models : fallbackModels();
}

function parseNousJson<T>(text: string, context: string): T {
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(`Nous ${context} returned a non-JSON response: ${text.slice(0, 200)}`);
  }
}

// ── Rate-limit handling ───────────────────────────────────────────────────
// The Nous inference API answers 429 with a JSON body shaped like:
//   { status: 429, message: string, reason: string, retry_after: number, alternates: string[] }
// We surface the server-suggested alternates and retry delay to the user, and
// return a non-retryable assistant error so the agent loop stops retrying the
// same model and the user can switch models instead of hammering the server.

interface RateLimitErrorBody {
  status?: number;
  message?: string;
  reason?: string;
  retry_after?: number | string;
  alternates?: string[];
}

function isRateLimitedErrorMessage(errorMessage: string): boolean {
  return /\b429\b|rate.?limit|rate_limited|too many requests|overloaded|at capacity|servers are currently busy|currently experiencing high demand/i.test(
    errorMessage,
  );
}

function parseRateLimitBody(errorMessage: string): RateLimitErrorBody | undefined {
  const start = errorMessage.lastIndexOf('{');
  if (start === -1) return undefined;
  try {
    return JSON.parse(errorMessage.slice(start)) as RateLimitErrorBody;
  } catch {
    return undefined;
  }
}

function buildRateLimitedAssistantMessage(
  model: Model<Api>,
  body: RateLimitErrorBody,
): AssistantMessage {
  const alternates =
    body.alternates && body.alternates.length ? body.alternates : ['a different Nous model'];
  const alternateText = alternates.join(', ');
  const retryAfter = body.retry_after;
  const retryText =
    typeof retryAfter === 'number' ? ` Try again in ${retryAfter}s if you prefer.` : '';
  const errorMessage = `Nous is refusing requests for this model due to a fair-use cap. Stop retrying; switch to ${alternateText}. Server alternates: ${alternateText}.${retryText}`;
  return {
    role: 'assistant',
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    stopReason: 'error',
    errorMessage,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    timestamp: Date.now(),
  };
}

function buildModelForAlternate(id: string): Model<'openai-completions'> {
  const m = modelMeta(id);
  return {
    id,
    name: m.name,
    api: 'openai-completions',
    provider: 'nous',
    baseUrl: NOUS_INFERENCE_URL,
    input: m.input,
    cost: m.cost,
    reasoning: m.reasoning,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    compat: m.compat,
  } as Model<'openai-completions'>;
}

function pickFreeAlternates(
  alternates: string[] | undefined,
  currentModelId: string,
): Model<'openai-completions'>[] {
  const list = alternates && alternates.length ? alternates : ['a different Nous model'];
  const freeAlternates: Model<'openai-completions'>[] = [];
  for (const alt of list) {
    if (typeof alt === 'string' && alt.endsWith(':free')) {
      const altId = alt.replace(/:free$/i, '');
      if (altId.toLowerCase() !== currentModelId.toLowerCase()) {
        freeAlternates.push(buildModelForAlternate(alt));
      }
    }
  }
  return freeAlternates;
}

function wrapOpenAIStream(
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const inner = openaiCompletionsStream(model as Model<'openai-completions'>, context, options);
  const outer = createAssistantMessageEventStream();
  let intercepted = false;

  (async () => {
    for await (const event of inner) {
      if (intercepted) {
        break;
      }
      if (
        event.type === 'error' &&
        event.reason === 'error' &&
        event.error?.stopReason === 'error' &&
        isRateLimitedErrorMessage(event.error?.errorMessage ?? '')
      ) {
        const errorMessage = event.error?.errorMessage;
        if (errorMessage !== undefined) {
          const rateLimit = parseRateLimitBody(errorMessage);
          if (rateLimit) {
            if (autoSwitchEnabled === true) {
              const currentModelId = model.id;
              const freeAlternates = pickFreeAlternates(rateLimit.alternates, currentModelId);
              if (freeAlternates.length > 0) {
                const alternate = freeAlternates[0]!;
                intercepted = true;
                const newStream = openaiCompletionsStream(alternate, context, options);
                for await (const newEvent of newStream) {
                  outer.push(newEvent);
                }
                return;
              }
            }
            intercepted = true;
            outer.end(buildRateLimitedAssistantMessage(model, rateLimit));
            continue;
          }
        }
      }
      outer.push(event);
    }
  })();

  return outer;
}

// ── Debug helpers ──
// Log all HTTP response bodies and JWT payloads during the OAuth flow.
// Set DEBUG_NOUS=1 in the environment to enable.

const DEBUG_LOGS: string[] = [];

function debugLog(...args: unknown[]): void {
  if (process.env['DEBUG_NOUS'] === '1') {
    const entry = args
      .map((a) => {
        if (typeof a === 'string') return a;
        return String(a);
      })
      .join(' ');
    DEBUG_LOGS.push(entry);
    console.warn('[NOUS-DEBUG]', entry);
  }
}

async function requestDeviceCode(): Promise<NousDeviceCodeResponse> {
  const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/device/code`, {
    method: 'POST',
    headers: {
      ...authHeaders(),
      'Content-Type': 'application/x-www-form-urlencoded',
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
    if (
      response.status === 429 &&
      (lower.includes('vercel security checkpoint') || lower.includes('verifying your browser'))
    ) {
      throw new Error('NOUS_VERCEL_CHECKPOINT');
    }
    throw new Error(`Nous device code request failed: ${response.status} ${text}`);
  }

  const data = parseNousJson<NousDeviceCodeResponse>(text, 'device code request');
  if (!data.device_code || !data.user_code || !data.verification_uri) {
    throw new Error('Nous device code response missing required fields');
  }
  return data;
}

async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error('Login cancelled'));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(t);
        reject(new Error('Login cancelled'));
      },
      { once: true },
    );
  });
}

async function pollForToken(
  device: NousDeviceCodeResponse,
  signal?: AbortSignal,
): Promise<NousTokenResponse> {
  const deadline = nowMs() + device.expires_in * 1000;
  let intervalMs = Math.max(
    1000,
    Math.min((device.interval || 5) * 1000, DEVICE_POLL_INTERVAL_CAP_MS),
  );

  while (nowMs() < deadline) {
    if (signal?.aborted) throw new Error('Login cancelled');

    const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/token`, {
      method: 'POST',
      headers: {
        ...authHeaders(),
        'Content-Type': 'application/x-www-form-urlencoded',
        Origin: NOUS_PORTAL_URL,
        Referer: `${NOUS_PORTAL_URL}/`,
      },
      body: new URLSearchParams({
        grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
        client_id: NOUS_CLIENT_ID,
        device_code: device.device_code,
      }).toString(),
    });

    const text = await response.text();
    let data: NousTokenResponse | null;
    try {
      data = text ? (JSON.parse(text) as NousTokenResponse) : null;
    } catch {
      data = null;
    }

    if (response.ok && data?.access_token) {
      debugLog(
        `[pollForToken] user_code=${device.user_code} HTTP ${response.status} got access_token`,
      );
      return data;
    }

    const error = data?.error;
    debugLog(
      `[pollForToken] user_code=${device.user_code} HTTP ${response.status} error=${error} body=${text.substring(0, 200)}`,
    );

    if (error === 'authorization_pending') {
      await sleep(intervalMs, signal);
      continue;
    }
    if (error === 'slow_down') {
      intervalMs = Math.min(intervalMs + 5000, 10000);
      await sleep(intervalMs, signal);
      continue;
    }
    if (error === 'expired_token') {
      throw new Error('Nous device code expired. Please try /login again.');
    }
    if (error === 'access_denied') {
      throw new Error('Nous authorization was denied.');
    }

    throw new Error(`Nous token request failed: ${response.status} ${text}`);
  }

  throw new Error('Nous login timed out.');
}

async function refreshAccessToken(credentials: OAuthCredentials): Promise<NousCredentials> {
  const refreshToken = getRefreshToken(credentials);
  if (!refreshToken) {
    debugLog('[refreshAccessToken] NO refresh token available');
    throw new Error('No Nous refresh token available');
  }

  debugLog('[refreshAccessToken] starting token refresh');
  const response = await fetch(`${NOUS_PORTAL_URL}/api/oauth/token`, {
    method: 'POST',
    headers: {
      ...authHeaders(),
      'Content-Type': 'application/x-www-form-urlencoded',
      Origin: NOUS_PORTAL_URL,
      Referer: `${NOUS_PORTAL_URL}/`,
    },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      client_id: NOUS_CLIENT_ID,
      refresh_token: refreshToken,
    }).toString(),
  });

  const text = await response.text();
  debugLog(`[refreshAccessToken] HTTP ${response.status} body=${text.substring(0, 300)}`);

  if (!response.ok) {
    throw new Error(`Nous token refresh failed: ${response.status} ${text}`);
  }

  const data = parseNousJson<NousTokenResponse>(text, 'token refresh');
  if (!data.access_token) {
    throw new Error('Nous token refresh succeeded without access_token');
  }

  const oauthExpiresAt = nowMs() + (data.expires_in || 3600) * 1000;
  return {
    refresh: data.refresh_token || refreshToken,
    access: data.access_token,
    expires: oauthExpiresAt,
    enterpriseUrl: safeBaseUrl(
      data.inference_base_url || (credentials as NousCredentials).enterpriseUrl,
    ),
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
    debugLog('[mintAgentKey] token expiring, refreshing');
    baseCreds = await refreshAccessToken(credentials);
  }

  const accessToken = getOAuthAccessToken(baseCreds) || baseCreds.access;
  if (!accessToken) throw new Error('No access token available for agent key minting');

  // The Portal's access_token IS the invoke JWT (nous_jwt_mining.md).
  // The Portal has retired the /api/oauth/agent-key minting endpoint (HTTP 410).
  // The access_token is stored directly as agent_key — no separate mint needed.
  const existingKey = getAgentKey(baseCreds);
  const existingKeyExpiry = getAgentKeyExpiryMs(baseCreds);

  if (existingKey && existingKeyExpiry && !isExpiring(existingKeyExpiry ?? undefined, 60 * 1000)) {
    debugLog('[mintAgentKey] using existing cached key');
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
        agentKeyExpiresAt: existingKeyExpiry ? existingKeyExpiry.toString() : undefined,
        keyId: baseCreds.metadata?.keyId,
      },
    };
  }

  debugLog('[mintAgentKey] access_token directly = agent_key (no minting endpoint)');
  debugLog('[mintAgentKey] access_token: ' + accessToken.substring(0, 100) + '...');
  debugLog('[mintAgentKey] scope: ' + getScope(baseCreds));
  debugLog('[mintAgentKey] expires: ' + baseCreds.expires);

  return {
    ...baseCreds,
    access: accessToken,
    enterpriseUrl: safeBaseUrl(baseCreds.enterpriseUrl),
    metadata: {
      ...baseCreds.metadata,
      refreshToken: getRefreshToken(baseCreds),
      tokenType: getTokenType(baseCreds),
      scope: getScope(baseCreds),
      oauthAccessToken: getOAuthAccessToken(baseCreds),
      oauthAccessExpiresAt: baseCreds.expires,
      agentKey: accessToken,
      agentKeyExpiresAt: getJwtExpMs(accessToken)?.toString(),
      keyId: baseCreds.metadata?.keyId,
    },
  };
}

async function loginNous(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
  try {
    const device = await requestDeviceCode();

    callbacks.onAuth({
      url: device.verification_uri_complete || device.verification_uri,
    });
    await callbacks.onPrompt({
      message: `Open the URL in your browser and approve access. If prompted, enter this code: ${device.user_code}. Press Enter here after approving to continue polling.`,
    });

    const token = await pollForToken(device, callbacks.signal);
    if (!token.access_token) throw new Error('Nous login did not return an access token');

    const oauthExpiresAt = nowMs() + (token.expires_in || 3600) * 1000;
    const credentials: NousCredentials = {
      refresh: token.refresh_token || '',
      access: token.access_token,
      expires: oauthExpiresAt,
      enterpriseUrl: safeBaseUrl(token.inference_base_url),
      metadata: {
        refreshToken: token.refresh_token || '',
        tokenType: token.token_type || 'Bearer',
        scope: token.scope || NOUS_SCOPE,
        oauthAccessToken: token.access_token,
        oauthAccessExpiresAt: oauthExpiresAt,
      },
    };

    // Debug: log the raw JWT payload
    debugLog('[loginNous] raw access_token: ' + token.access_token.substring(0, 100) + '...');
    const payload = decodeJwtPayload(token.access_token);
    debugLog('[loginNous] JWT payload: ' + JSON.stringify(payload));
    debugLog('[loginNous] NOUS_SCOPE=' + NOUS_SCOPE);

    const minted = await mintAgentKey(credentials);
    const annotated = await annotateFreeTier(minted);
    debugLog(
      '[loginNous] final credentials.agentKey: ' +
        (annotated as NousCredentials).metadata?.agentKey?.substring(0, 100),
    );
    return {
      ...annotated,
      metadata: {
        ...annotated.metadata,
        oauthAccessToken: token.access_token,
        oauthAccessExpiresAt: oauthExpiresAt,
      },
    };
  } catch (error) {
    if (!(error instanceof Error) || error.message !== 'NOUS_VERCEL_CHECKPOINT') throw error;

    callbacks.onAuth({ url: `${NOUS_PORTAL_URL}/login` });
    const pasted = await callbacks.onPrompt({
      message:
        'Nous Portal is behind a Vercel browser checkpoint. Open the login page in your normal browser, complete login there, then paste a Nous inference agent key or access token:',
    });

    const token = pasted.trim();
    if (!token) throw new Error('No token provided', { cause: error });

    if (token.startsWith('sk-')) {
      return {
        refresh: '',
        access: token,
        expires: nowMs() + 24 * 60 * 60 * 1000,
        enterpriseUrl: safeBaseUrl(NOUS_INFERENCE_URL),
        metadata: { tokenType: 'Bearer', scope: NOUS_SCOPE, agentKey: token },
      };
    }

    const creds: NousCredentials = {
      refresh: '',
      access: token,
      expires: nowMs() + 60 * 60 * 1000,
      enterpriseUrl: safeBaseUrl(NOUS_INFERENCE_URL),
      metadata: { tokenType: 'Bearer', scope: NOUS_SCOPE },
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
    const envApiKey = process.env['NOUS_API_KEY']?.trim();
    if (envApiKey) discoveredModels = await fetchLiveModels(envApiKey);
    else discoveredModels = await resolveStartupNousModels();
  } catch {
    // Keep fallback models. Runtime OAuth flow is the primary path.
  }

  void pi.registerProvider('nous', {
    baseUrl: NOUS_INFERENCE_URL,
    api: 'openai-completions',
    streamSimple: wrapOpenAIStream,
    models: discoveredModels,
    oauth: {
      name: 'Nous Portal',
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
            ? (models as { models: typeof discoveredModels }).models
            : discoveredModels;
        const c = credentials as NousCredentials;
        const baseUrl = safeBaseUrl(c.enterpriseUrl);
        const freeTier = c.metadata?.freeTier === true;

        return currentModels.flatMap((model) => {
          const m = model as NousModelConfig;
          if (m.provider !== 'nous') return [m];
          if (freeTier && !isModelFree(m)) return [];
          return [{ ...m, baseUrl }];
        });
      },
    },
  });

  pi.registerFlag('nous-auto-switch', {
    description:
      'Automatically switch to a free alternate model when the current model hits a fair-use cap (429).',
    type: 'boolean',
    default: false,
  });

  autoSwitchEnabled = pi.getFlag('nous-auto-switch') === true;

  void pi.registerCommand('nous-switch-model', {
    description: 'List available Nous models and switch to an alternate.',
    handler: async (_args, ctx) => {
      const creds = await loadPersistedNousCredentials();
      if (!creds?.access) {
        ctx.ui.notify('Nous login first (run /login or set NOUS_API_KEY).', 'error');
        return;
      }
      const agentKey = creds.metadata?.agentKey || creds.access;
      let models: NousModel[];
      try {
        models = await fetchLiveModels(agentKey, NOUS_INFERENCE_URL);
      } catch (error) {
        ctx.ui.notify(`Could not fetch Nous models: ${(error as Error).message}`, 'error');
        return;
      }
      const ids = models.map((model) => model.id);
      const index = await ctx.ui.select(`Nous models (${ids.length})`, ids);
      if (index === undefined) return;
      const chosenModel = models.find((model) => model.id === index);
      if (chosenModel === undefined) return;
      ctx.ui.notify(`Nous model: ${chosenModel.id}`, 'info');
    },
  });

  void pi.registerCommand('nous-debug', {
    description:
      'Probe the Nous API: resolve credentials, fetch /v1/models and /api/oauth/account, report auth plus entitlement. With "on"/"off" enables/disables debug tracing.',
    handler: async (args, ctx) => {
      const out: string[] = [];
      const ok = (s: string) => out.push(`  ok   ${s}`);
      const bad = (s: string) => out.push(`  failed ${s}`);

      const start = (s: string) => out.push(`--- ${s} ---`);
      const summary = (passed: boolean, failedCount: number) => {
        out.push(`DIAGNOSTICS: ${passed ? 'PASS' : `FAIL (\${failedCount} check(s) failed)`}`);
      };

      const trimmed = args.trim().toLowerCase();

      // Control debug tracing
      if (trimmed === 'on' || trimmed === 'enable' || trimmed === 'true') {
        process.env['DEBUG_NOUS'] = '1';
        DEBUG_LOGS.length = 0;
        ctx.ui.notify('Nous debug tracing enabled', 'info');
        return;
      }
      if (trimmed === 'off' || trimmed === 'disable' || trimmed === 'false') {
        delete process.env['DEBUG_NOUS'];
        DEBUG_LOGS.length = 0;
        ctx.ui.notify('Nous debug tracing disabled', 'info');
        return;
      }

      // Show Nous API diagnostics
      ctx.ui.notify('Nous API diagnostics', 'info');

      let apiKey: string | undefined = process.env['NOUS_API_KEY']?.trim();
      if (apiKey) {
        ok('NOUS_API_KEY env is set');
      } else {
        start('Authentication');
        const creds = await loadPersistedNousCredentials();
        if (!creds || !creds.access) {
          bad('no Nous credentials found (set NOUS_API_KEY or run a Nous login first)');
          summary(false, 1);
          ctx.ui.notify(out.join('\n'), 'error');
          return;
        }
        apiKey = creds.access;
        ok('agent key loaded from ~/.pi/agent/auth.json');
      }

      let oauthToken: string | undefined;
      const creds = await loadPersistedNousCredentials();
      if (creds?.metadata?.oauthAccessToken) {
        oauthToken = creds.metadata.oauthAccessToken;
        ok('OAuth token loaded from ~/.pi/agent/auth.json');
      } else {
        bad('no OAuth token in ~/.pi/agent/auth.json');
      }

      start('Endpoints');
      ok('models endpoint: GET ' + NOUS_INFERENCE_URL + '/models');
      try {
        const res = await fetch(NOUS_INFERENCE_URL + '/models', { headers: authHeaders(apiKey) });
        const payload = (await res.json()) as { data?: { id?: string }[]; error?: string };
        if (res.ok) {
          const ids = (payload.data ?? [])
            .map((m) => m.id ?? '')
            .filter(Boolean)
            .slice(0, 5);
          ok(
            `HTTP ${res.status}; ${payload.data?.length ?? 0} model(s); first ids: ${ids.join(' ')}`,
          );
        } else {
          bad(`HTTP ${res.status}; error: ${(payload.error as string) ?? 'unknown'}`);
        }
      } catch (e) {
        bad(`request failed: \$(e as Error).message`);
      }

      if (oauthToken) {
        ok('account endpoint: GET ' + NOUS_PORTAL_URL + '/api/oauth/account');
        try {
          const res = await fetch(NOUS_PORTAL_URL + '/api/oauth/account', {
            headers: authHeaders(oauthToken),
          });
          const body = (await res.json()) as Record<string, unknown>;
          if (res.ok) {
            ok(`HTTP ${res.status}; subscription=${JSON.stringify(body['subscription'])}`);
            const credits = body['purchased_credits_remaining'];
            ok(`purchased_credits_remaining=${typeof credits === 'number' ? credits : 'n/a'}`);
            const toolAccess = body['tool_access'] as Record<string, unknown> | undefined;
            ok(`tool_access.enabled=${toolAccess?.['enabled'] ?? 'n/a'}`);
            const paid = body['paid_service_access'] as Record<string, unknown> | undefined;
            ok(`paid_service_access.allowed=${paid?.['allowed'] ?? 'n/a'}`);
            if (!paid?.['allowed']) {
              ok('no credits on account -> inference API calls will be blocked');
            }
          } else {
            bad(`HTTP ${res.status}; body=\${JSON.stringify(body)}`);
          }
        } catch (e) {
          bad(`request failed: \$(e as Error).message`);
        }
      } else {
        bad('skipped account endpoint (no OAuth token)');
      }

      const failedCount = out.filter((l) => l.startsWith('  failed')).length;
      const passed = failedCount === 0;
      summary(passed, failedCount);

      // Append accumulated debug logs (if DEBUG_NOUS=1 was set during a previous login)
      if (DEBUG_LOGS.length > 0) {
        out.push('');
        out.push('--- debug logging (DEBUG_NOUS=1) ---');
        out.push(...DEBUG_LOGS);
        out.push('--- end debug logging ---');
      } else {
        out.push('');
        out.push('--- debug logging ---');
        out.push(
          'No debug logs accumulated. Run the login with DEBUG_NOUS=1 to capture debug output.',
        );
        out.push('--- end debug logging ---');
      }

      ctx.ui.notify(out.join('\n'), passed ? 'info' : 'error');
    },
  });
}
