// MemeCatz edge worker. Serves the static site from ./web and proxies one
// endpoint, /api/giphy, so the Giphy key can live as a Cloudflare secret instead
// of inside browser JavaScript, where anyone could read it.
//
// The key is read from env.GIPHY_API_KEY. If it is not set, the endpoint
// answers 404 and the page falls back to the keyless photo sources.

const GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search";
const MAX_LIMIT = 12;

function json(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", ...extraHeaders },
  });
}

export async function handleGiphy(request, env) {
  const key = env.GIPHY_API_KEY;
  if (!key) return json({ error: "giphy not configured" }, 404);

  const url = new URL(request.url);
  const q = (url.searchParams.get("q") || "").trim().slice(0, 80);
  if (!q) return json({ error: "missing q" }, 400);

  let limit = parseInt(url.searchParams.get("limit") || "8", 10);
  if (!Number.isFinite(limit) || limit < 1) limit = 8;
  if (limit > MAX_LIMIT) limit = MAX_LIMIT;

  const upstream = new URL(GIPHY_SEARCH_URL);
  upstream.searchParams.set("api_key", key);
  upstream.searchParams.set("q", q);
  upstream.searchParams.set("limit", String(limit));
  upstream.searchParams.set("rating", "g");
  upstream.searchParams.set("lang", "en");

  let res;
  try {
    res = await fetch(upstream.toString(), { cf: { cacheTtl: 300, cacheEverything: true } });
  } catch (err) {
    return json({ error: "upstream unreachable" }, 502);
  }
  if (!res.ok) return json({ error: `giphy ${res.status}` }, 502);

  const payload = await res.json();
  // Strip the response down to what the page needs. Never forward the key.
  const items = (payload.data || [])
    .map((item) => {
      const img = item.images || {};
      const animated = img.fixed_height || img.downsized || img.original || {};
      const still = img.fixed_height_still || img.original_still || {};
      return item.id && animated.url
        ? { id: item.id, url: animated.url, still: still.url || null, title: item.title || "" }
        : null;
    })
    .filter(Boolean);

  return json({ items }, 200, { "cache-control": "public, max-age=300" });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/giphy") return handleGiphy(request, env);
    return env.ASSETS.fetch(request);
  },
};
