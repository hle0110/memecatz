// Real reactions from three sources, in order of preference:
//
//   1. Giphy, mood matched, through the /api/giphy proxy on this same site. The
//      key lives as a Cloudflare secret, never in this file. This is where the
//      actual meme reactions come from.
//   2. The Cat API or Dog CEO API directly. Real photos, not mood matched, no
//      key needed. Used when the proxy reports no key is configured.
//
// If everything fails the page shows an honest waiting state rather than
// substituting a fake image. All direct calls are plain GETs with no custom
// headers, which avoids a CORS preflight.

const GIPHY_PROXY = "/api/giphy";
const CAT_API_SEARCH_URL = "https://api.thecatapi.com/v1/images/search";
const DOG_API_URL = "https://dog.ceo/api/breeds/image/random";
const BATCH_SIZE = 8;
const RETRY_COOLDOWN_MS = 30000;
const MIN_POOL = 3;

// Ported from reactions.py. Each mood maps to a search phrase; {animal} is
// filled with cat or dog.
export const MOOD_QUERY_TEMPLATES = {
  happy: "happy {animal}",
  sad: "sad {animal}",
  angry: "angry {animal}",
  surprise: "surprised {animal}",
  fear: "scared {animal}",
  disgust: "disgusted {animal}",
  neutral: "{animal} staring blankly",
  smug: "smug {animal}",
  confused: "confused {animal}",
  mischief: "mischievous {animal}",
  annoyed: "annoyed {animal}",
  approval: "{animal} nodding approval",
  disapproval: "{animal} side eye disapproval",
  chill: "relaxed {animal}",
  suspicious: "suspicious {animal}",
  triumph: "{animal} victory",
  anxious: "nervous {animal}",
  bored: "bored {animal}",
  mocking: "sassy {animal}",
  stop: "{animal} stop paw",
  determined: "determined {animal}",
  focused: "focused {animal}",
};

export function queryForMood(moodTag, animal = "cat") {
  const template = MOOD_QUERY_TEMPLATES[moodTag];
  if (!template) return `${moodTag.replace(/_/g, " ")} ${animal}`;
  return template.replace("{animal}", animal);
}

export class ReactionSource {
  constructor(animal = "cat") {
    this.animal = animal === "dog" ? "dog" : "cat";
    this.giphyAvailable = null; // unknown until the first proxy call
    this.moodPools = {};        // mood -> entries from giphy
    this.generalPool = [];      // keyless photos
    this.failedAt = {};         // source key -> timestamp
    this.inflight = new Set();
  }

  setAnimal(animal) {
    const next = animal === "dog" ? "dog" : "cat";
    if (next !== this.animal) {
      this.animal = next;
      this.moodPools = {};
      this.generalPool = [];
      this.failedAt = {};
    }
  }

  describeSource() {
    if (this.giphyAvailable) return `mood matched ${this.animal} reactions via Giphy`;
    return this.animal === "dog"
      ? "real dog photos via Dog CEO API"
      : "real cat photos via The Cat API";
  }

  _cooling(key) {
    return Date.now() - (this.failedAt[key] || 0) < RETRY_COOLDOWN_MS;
  }

  async _fetchGiphy(mood) {
    const key = `giphy:${mood}`;
    if (this.giphyAvailable === false || this._cooling(key) || this.inflight.has(key)) return;
    this.inflight.add(key);
    try {
      const q = encodeURIComponent(queryForMood(mood, this.animal));
      const res = await fetch(`${GIPHY_PROXY}?q=${q}&limit=${BATCH_SIZE}`);
      if (res.status === 404) {
        // Proxy says no key is configured. Stop asking.
        this.giphyAvailable = false;
        return;
      }
      if (!res.ok) throw new Error(`giphy proxy ${res.status}`);
      const payload = await res.json();
      const items = (payload.items || []).map((it) => ({
        key: `giphy:${it.id}`,
        url: it.url,
        name: `${mood} ${this.animal} reaction`,
        attribution: "Powered By GIPHY",
      }));
      if (items.length) {
        this.giphyAvailable = true;
        this.moodPools[mood] = [...(this.moodPools[mood] || []), ...items];
      } else {
        this.failedAt[key] = Date.now();
      }
    } catch (err) {
      this.failedAt[key] = Date.now();
      console.warn("giphy fetch failed, will retry later:", err.message);
    } finally {
      this.inflight.delete(key);
    }
  }

  async _fetchGeneral() {
    const key = "general";
    if (this._cooling(key) || this.inflight.has(key)) return;
    this.inflight.add(key);
    try {
      let items = [];
      if (this.animal === "dog") {
        for (let i = 0; i < BATCH_SIZE; i++) {
          const res = await fetch(DOG_API_URL);
          if (!res.ok) throw new Error(`dog api ${res.status}`);
          const p = await res.json();
          if (p.status !== "success" || !p.message) throw new Error("unexpected dog api payload");
          items.push({ key: `dogapi:${p.message}`, url: p.message, name: "real dog photo", attribution: null });
        }
      } else {
        const res = await fetch(`${CAT_API_SEARCH_URL}?limit=${BATCH_SIZE}`);
        if (!res.ok) throw new Error(`cat api ${res.status}`);
        const p = await res.json();
        if (!Array.isArray(p)) throw new Error("unexpected cat api payload");
        items = p
          .filter((it) => it && it.id && it.url)
          .map((it) => ({ key: `catapi:${it.id}`, url: it.url, name: "real cat photo", attribution: null }));
      }
      if (items.length) this.generalPool.push(...items);
      else this.failedAt[key] = Date.now();
    } catch (err) {
      this.failedAt[key] = Date.now();
      console.warn("photo fetch failed, will retry later:", err.message);
    } finally {
      this.inflight.delete(key);
    }
  }

  // Kick off fetches for the given moods without blocking.
  warm(moods) {
    for (const m of moods) {
      if ((this.moodPools[m] || []).length < MIN_POOL) this._fetchGiphy(m);
    }
    if (this.generalPool.length < MIN_POOL) this._fetchGeneral();
  }

  // Returns an entry matched to the strongest mood that has one, else a general
  // photo, else null while everything is still loading.
  pick(moods, excludeKey = null) {
    this.warm(moods);

    for (const m of moods) {
      const pool = this.moodPools[m] || [];
      const chosen = this._takeFrom(pool, excludeKey);
      if (chosen) {
        this.moodPools[m] = pool.filter((e) => e.key !== chosen.key);
        return chosen;
      }
    }

    const chosen = this._takeFrom(this.generalPool, excludeKey);
    if (chosen) this.generalPool = this.generalPool.filter((e) => e.key !== chosen.key);
    return chosen;
  }

  _takeFrom(pool, excludeKey) {
    if (!pool.length) return null;
    let candidates = pool.filter((e) => e.key !== excludeKey);
    if (!candidates.length) candidates = pool;
    return candidates[Math.floor(Math.random() * candidates.length)];
  }
}
