import { Buffer } from "node:buffer";
import process from "node:process";

export const config = {
  api: {
    bodyParser: false,
  },
};

const DEFAULT_TIMEOUT_MS = 600000; // 10 minutes for async Textract processing
const BODYLESS_METHODS = new Set(["GET", "HEAD"]);
const SKIPPED_HEADERS = new Set(["host", "content-length", "x-api-key"]);

const stripTrailingSlash = (value) => String(value || "").replace(/\/+$/, "");

const readBody = async (req) => {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
};

const getPathPartsFromQuery = (query) => {
  const fromQuery = Array.isArray(query?.path)
    ? query.path
    : [query?.path].filter(Boolean);
  return fromQuery[0] === "api" ? fromQuery.slice(1) : fromQuery;
};

const getPathPartsFromUrl = (requestUrl) => {
  const parsed = new URL(requestUrl || "/", "http://localhost");
  const rawPath = parsed.pathname.replace(/^\/+/, "");
  const normalized = rawPath.startsWith("api/") ? rawPath.slice(4) : rawPath;
  return normalized ? normalized.split("/").filter(Boolean) : [];
};

const resolvePathParts = (req) => {
  const queryPath = getPathPartsFromQuery(req.query);
  return queryPath.length > 0 ? queryPath : getPathPartsFromUrl(req.url);
};

const buildUpstreamUrl = (backendBase, req) => {
  const pathParts = resolvePathParts(req);
  const upstream = new URL(`${backendBase}/${pathParts.join("/")}`);

  for (const [key, value] of Object.entries(req.query || {})) {
    if (key === "path" || value == null) continue;
    if (Array.isArray(value)) {
      for (const item of value) upstream.searchParams.append(key, String(item));
      continue;
    }
    upstream.searchParams.append(key, String(value));
  }

  return upstream;
};

const buildUpstreamHeaders = (incomingHeaders, apiKey) => {
  const headers = new Headers();

  for (const [key, value] of Object.entries(incomingHeaders || {})) {
    if (!value) continue;
    if (SKIPPED_HEADERS.has(key.toLowerCase())) continue;
    headers.set(key, Array.isArray(value) ? value.join(",") : String(value));
  }

  if (apiKey) headers.set("X-API-KEY", apiKey);
  return headers;
};

const readRequestBody = async (req, method) => {
  if (BODYLESS_METHODS.has(method)) return undefined;
  return readBody(req);
};

const fetchWithTimeout = async (url, options, timeoutMs) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
};

const sendUpstreamResponse = async (res, response) => {
  const contentType = response.headers.get("content-type") || "application/json";
  const payload = Buffer.from(await response.arrayBuffer());
  res.setHeader("content-type", contentType);
  res.setHeader("cache-control", "no-store");
  return res.status(response.status).send(payload);
};

export default async function handler(req, res) {
  const backendBase = stripTrailingSlash(process.env.OCR_BACKEND_URL);
  const apiKey = process.env.OCR_API_KEY;

  if (!backendBase) {
    return res
      .status(500)
      .json({ detail: "Server misconfiguration: OCR_BACKEND_URL is not set" });
  }

  const upstream = buildUpstreamUrl(backendBase, req);
  const headers = buildUpstreamHeaders(req.headers, apiKey);

  const method = req.method || "GET";
  const body = await readRequestBody(req, method);

  try {
    const response = await fetchWithTimeout(
      upstream.toString(),
      {
        method,
        headers,
        body,
      },
      DEFAULT_TIMEOUT_MS,
    );
    return sendUpstreamResponse(res, response);
  } catch (error) {
    const detail =
      error?.name === "AbortError"
        ? "Upstream OCR API timeout"
        : "Failed to reach OCR API";
    return res.status(502).json({ detail });
  }
}
