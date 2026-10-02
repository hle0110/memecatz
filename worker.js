// MemeCatz edge worker. Cloudflare serves the files in ./web directly, with
// security headers from web/_headers. This script only runs for requests that
// do not match a file, and answers one small endpoint:
//
//   /api/config   tells the page which optional services are configured.
//
// Giphy requires search requests to be made directly from the browser, not
// through a proxy, so the page calls Giphy itself with the key it gets here.
// The key lives as a Cloudflare secret named GIPHY_API_KEY so it can be
// rotated without a code change. If it is not set, the page uses the keyless
// photo sources instead.

// Same policy as web/_headers (tests/test_web_parity.py checks they match).
// Lists every host the page is allowed to load from or talk to.
const CSP = [
  "default-src 'self'",
  "script-src 'self' https://cdn.jsdelivr.net 'wasm-unsafe-eval'",
  "connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com https://api.giphy.com https://api.thecatapi.com https://dog.ceo",
  "img-src 'self' data: blob: https://*.giphy.com https://s3.us-west-2.amazonaws.com https://cdn2.thecatapi.com https://images.dog.ceo",
  "media-src 'self' blob:",
  "style-src 'self'",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join("; ");

const SECURITY_HEADERS = {
  "content-security-policy": CSP,
  "permissions-policy": "camera=(self), microphone=(), geolocation=(), payment=(), usb=()",
  "referrer-policy": "strict-origin-when-cross-origin",
  "x-content-type-options": "nosniff",
};

function withHeaders(response, extra = {}) {
  const out = new Response(response.body, response);
  for (const [k, v] of Object.entries({ ...SECURITY_HEADERS, ...extra })) out.headers.set(k, v);
  return out;
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function handleConfig(env) {
  const giphyKey = typeof env.GIPHY_API_KEY === "string" && env.GIPHY_API_KEY.trim() ? env.GIPHY_API_KEY.trim() : null;
  return json({ giphyKey });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/config") {
      if (request.method !== "GET" && request.method !== "HEAD") return withHeaders(json({ error: "method not allowed" }, 405));
      return withHeaders(handleConfig(env));
    }
    if (url.pathname.startsWith("/api/")) return withHeaders(json({ error: "not found" }, 404));
    return withHeaders(await env.ASSETS.fetch(request));
  },
};
