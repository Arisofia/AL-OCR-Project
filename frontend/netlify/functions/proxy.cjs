const DEFAULT_TIMEOUT_MS = 30000;
const fs = require("node:fs");
const path = require("node:path");

const stripTrailingSlash = (value) => String(value || "").replace(/\/+$/, "");

const readRuntimeConfig = () => {
  try {
    const file = path.join(__dirname, "proxy.runtime.json");
    if (!fs.existsSync(file)) return {};
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return {};
  }
};

exports.handler = async (event) => {
  const runtimeConfig = readRuntimeConfig();
  const backendBase = stripTrailingSlash(
    process.env.OCR_BACKEND_URL || runtimeConfig.OCR_BACKEND_URL
  );
  const apiKey = process.env.OCR_API_KEY || runtimeConfig.OCR_API_KEY;

  if (!backendBase) {
    return {
      statusCode: 500,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        detail: "Server misconfiguration: OCR_BACKEND_URL is not set",
      }),
    };
  }

  const splat = event?.pathParameters?.splat || "";
  const upstreamPath = `/${String(splat).replace(/^\/+/, "")}`;
  const query =
    event.rawQuery ||
    new URLSearchParams(event.queryStringParameters || {}).toString();
  const querySuffix = query ? `?${query}` : "";
  const upstreamUrl = `${backendBase}${upstreamPath}${querySuffix}`;

  const forwardHeaders = new Headers();
  for (const [key, value] of Object.entries(event.headers || {})) {
    if (typeof value !== "string" || value.length === 0) continue;
    const lower = key.toLowerCase();
    if (["host", "content-length", "x-api-key"].includes(lower)) continue;
    forwardHeaders.set(key, value);
  }

  if (apiKey) {
    forwardHeaders.set("X-API-KEY", apiKey);
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  const httpMethod = event.httpMethod || "GET";
  const requestBody = event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64")
    : event.body;
  const fetchBody = ["GET", "HEAD"].includes(httpMethod)
    ? undefined
    : requestBody;

  try {
    const response = await fetch(upstreamUrl, {
      method: httpMethod,
      headers: forwardHeaders,
      body: fetchBody,
      signal: controller.signal,
    });

    const text = await response.text();
    const contentType =
      response.headers.get("content-type") || "application/json";

    return {
      statusCode: response.status,
      headers: {
        "content-type": contentType,
        "cache-control": "no-store",
      },
      body: text,
    };
  } catch (error) {
    const detail =
      error?.name === "AbortError"
        ? "Upstream OCR API timeout"
        : "Failed to reach OCR API";

    return {
      statusCode: 502,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ detail }),
    };
  } finally {
    clearTimeout(timeout);
  }
};
