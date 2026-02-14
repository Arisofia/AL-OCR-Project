import { Buffer } from "node:buffer";
import process from "node:process";

export const config = {
  api: {
    bodyParser: false,
  },
};

const DEFAULT_TIMEOUT_MS = 30000;

const stripTrailingSlash = (value) => String(value || "").replace(/\/+$/, "");

const readBody = async (req) => {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
};

export default async function handler(req, res) {
  const backendBase = stripTrailingSlash(process.env.OCR_BACKEND_URL);
  const apiKey = process.env.OCR_API_KEY;

  if (!backendBase) {
    return res
      .status(500)
      .json({ detail: "Server misconfiguration: OCR_BACKEND_URL is not set" });
  }

  const pathParts = Array.isArray(req.query.path)
    ? req.query.path
    : [req.query.path].filter(Boolean);
  const upstreamPath = pathParts.join("/");
  const upstream = new URL(`${backendBase}/${upstreamPath}`);

  for (const [key, value] of Object.entries(req.query)) {
    if (key === "path") continue;
    if (Array.isArray(value)) {
      for (const item of value) upstream.searchParams.append(key, String(item));
    } else if (value != null) {
      upstream.searchParams.append(key, String(value));
    }
  }

  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (!value) continue;
    const lower = key.toLowerCase();
    if (["host", "content-length", "x-api-key"].includes(lower)) continue;
    headers.set(key, Array.isArray(value) ? value.join(",") : String(value));
  }
  if (apiKey) headers.set("X-API-KEY", apiKey);

  const method = req.method || "GET";
  const body =
    method === "GET" || method === "HEAD" ? undefined : await readBody(req);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(upstream.toString(), {
      method,
      headers,
      body,
      signal: controller.signal,
    });

    const contentType =
      response.headers.get("content-type") || "application/json";
    const payload = Buffer.from(await response.arrayBuffer());

    res.setHeader("content-type", contentType);
    res.setHeader("cache-control", "no-store");
    return res.status(response.status).send(payload);
  } catch (error) {
    const detail =
      error && error.name === "AbortError"
        ? "Upstream OCR API timeout"
        : "Failed to reach OCR API";
    return res.status(502).json({ detail });
  } finally {
    clearTimeout(timeout);
  }
}
