// Real reactions from two kinds of source, in order of preference:
//
//   1. Giphy, mood matched. The page asks this site's /api/config for the
//      Giphy key, then calls Giphy directly from the browser, which is what
//      Giphy's integration rules require. Results are used in the order Giphy
//      returns them, nothing is filtered out or stored beyond this page visit.
//   2. The Cat API or Dog CEO API. Real photos, not mood matched, no key
//      needed. Used only when Giphy is not configured or not reachable, so the
//      two sources are never mixed while Giphy is working.
//
// If everything fails the page shows an honest waiting state rather than a
// substitute image. All requests are plain GETs with no custom headers, which
// avoids a CORS preflight.

const CONFIG_URL = "/api/config";
const GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search";
const CAT_API_SEARCH_URL = "https://api.thecatapi.com/v1/images/search";
const DOG_BREED_URL = (breed) => `https://dog.ceo/api/breed/${breed}/images/random/`;
const BATCH_SIZE = 8;
const RETRY_COOLDOWN_MS = 30000;
const MIN_POOL = 3;

// Pet breeds only. The Dog CEO "any breed" endpoint also returns wild canids
// such as African wild dogs and dholes, which are not what people expect.
// Every entry here was checked against https://dog.ceo/api/breeds/list/all.
export const DOG_BREEDS = [
  "beagle", "boxer", "bulldog/french", "cavapoo", "chihuahua", "cockapoo", "collie/border",
  "corgi/cardigan", "dachshund", "dalmatian", "frise/bichon", "german/shepherd", "havanese",
  "husky", "labrador", "malamute", "maltese", "papillon", "pembroke", "pomeranian",
  "poodle/toy", "pug", "retriever/golden", "samoyed", "shiba", "shihtzu",
];

// Same as MOOD_QUERY_TEMPLATES in reactions.py. Each mood maps to a search
// phrase; {animal} is filled with cat or dog.
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

// One Giphy result -> entry, or null if it has no usable image. The URL is
// used exactly as Giphy returns it. WebP is Giphy's smaller animated rendition
// and every current browser shows it in an <img>; the GIF is the fallback.
export function giphyEntry(item, mood, animal) {
  const fh = (item && item.images && item.images.fixed_height) || {};
  const url = fh.webp || fh.url;
  if (!item || !item.id || !url) return null;
  return {
    key: `giphy:${item.id}`,
    url,
    name: `${mood} ${animal} reaction`,
    attribution: "Powered By GIPHY",
    source: "giphy",
    cors: true,
  };
}

export class ReactionSource {
  constructor(animal = "cat", fetchImpl = (...args) => fetch(...args)) {
    this.fetch = fetchImpl;
    this.animal = animal === "dog" ? "dog" : "cat";
    this.giphyKey = undefined;     // undefined: not asked yet, null: not configured
    this.configPromise = null;
    this.configRetryAt = 0;
    this.giphyAvailable = null;    // unknown until the first search answers
    this.giphyTroubleAt = 0;       // last failed Giphy request (rate limit, outage)
    this._reset();
  }

  _reset() {
    this.moodQueues = {};          // mood -> entries in Giphy's order
    this.moodOffsets = {};         // mood -> next Giphy offset
    this.generalPool = [];         // keyless photos
    this.failedAt = {};
    this.inflight = new Set();
    this.seen = new Set();
  }

  setAnimal(animal) {
    const next = animal === "dog" ? "dog" : "cat";
    if (next !== this.animal) {
      this.animal = next;
      this._reset();
    }
  }

  describeSource() {
    if (this.giphyAvailable) return `mood matched ${this.animal} reactions via Giphy`;
    return this.animal === "dog" ? "real dog photos via Dog CEO API" : "real cat photos via The Cat API";
  }

  _cooling(key) {
    return Date.now() - (this.failedAt[key] || 0) < RETRY_COOLDOWN_MS;
  }

  // Asks this site whether Giphy is configured. A clear "no" (no key, or an
  // older deploy without the endpoint) is final for this visit. A network
  // error or server error is retried after the cooldown.
  _loadConfig() {
    if (this.configRetryAt && Date.now() >= this.configRetryAt) {
      this.configRetryAt = 0;
      this.configPromise = null;
    }
    if (!this.configPromise) {
      this.configPromise = (async () => {
        try {
          const res = await this.fetch(CONFIG_URL, { cache: "no-store" });
          if (res.status >= 500) throw new Error(`config ${res.status}`);
          if (!res.ok) {
            this.giphyKey = null;
          } else {
            const cfg = await res.json();
            this.giphyKey = typeof cfg.giphyKey === "string" && cfg.giphyKey ? cfg.giphyKey : null;
          }
          if (!this.giphyKey) this.giphyAvailable = false;
        } catch (err) {
          this.giphyKey = undefined;
          this.configRetryAt = Date.now() + RETRY_COOLDOWN_MS;
          console.warn("could not reach /api/config, using photos for now:", err.message);
        }
      })();
    }
    return this.configPromise;
  }

  async _fetchGiphy(mood) {
    const key = `giphy:${mood}`;
    if (this.giphyAvailable === false || this._cooling(key) || this.inflight.has(key)) return;
    this.inflight.add(key);
    try {
      await this._loadConfig();
      if (!this.giphyKey) return;
      const offset = this.moodOffsets[mood] || 0;
      const url = new URL(GIPHY_SEARCH_URL);
      url.searchParams.set("api_key", this.giphyKey);
      url.searchParams.set("q", queryForMood(mood, this.animal));
      url.searchParams.set("limit", String(BATCH_SIZE));
      url.searchParams.set("offset", String(offset));
      url.searchParams.set("rating", "g");
      url.searchParams.set("lang", "en");
      const res = await this.fetch(url.toString());
      if (res.status === 401 || res.status === 403) {
        // The key is wrong or revoked. Stop asking for this visit.
        this.giphyAvailable = false;
        console.warn(`giphy rejected the key (${res.status}), using photos instead`);
        return;
      }
      if (!res.ok) throw new Error(`giphy ${res.status}`);
      const payload = await res.json();
      const data = Array.isArray(payload.data) ? payload.data : [];
      const entries = data.map((it) => giphyEntry(it, mood, this.animal)).filter(Boolean);
      // Past the end of the results, start over from the top next time.
      this.moodOffsets[mood] = data.length < BATCH_SIZE ? 0 : offset + data.length;
      if (entries.length) {
        this.giphyAvailable = true;
        this.moodQueues[mood] = [...(this.moodQueues[mood] || []), ...entries];
      } else {
        this.failedAt[key] = Date.now();
      }
    } catch (err) {
      this.failedAt[key] = Date.now();
      this.giphyTroubleAt = Date.now();
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
        const breed = DOG_BREEDS[Math.floor(Math.random() * DOG_BREEDS.length)];
        const res = await this.fetch(DOG_BREED_URL(breed) + BATCH_SIZE);
        if (!res.ok) throw new Error(`dog api ${res.status}`);
        const p = await res.json();
        if (p.status !== "success" || !Array.isArray(p.message)) throw new Error("unexpected dog api payload");
        items = p.message.map((u) => ({
          key: `dogapi:${u}`, url: u, name: "real dog photo", attribution: null, source: "dog_api", cors: true,
        }));
      } else {
        const res = await this.fetch(`${CAT_API_SEARCH_URL}?limit=${BATCH_SIZE}`);
        if (!res.ok) throw new Error(`cat api ${res.status}`);
        const p = await res.json();
        if (!Array.isArray(p)) throw new Error("unexpected cat api payload");
        // The Cat API's image host sends no CORS header, so these photos can be
        // shown but not drawn into a saved snapshot.
        items = p
          .filter((it) => it && it.id && it.url)
          .map((it) => ({
            key: `catapi:${it.id}`, url: it.url, name: "real cat photo", attribution: null, source: "cat_api", cors: false,
          }));
      }
      items = items.filter((it) => !this.seen.has(it.key));
      if (items.length) this.generalPool.push(...items);
      else this.failedAt[key] = Date.now();
    } catch (err) {
      this.failedAt[key] = Date.now();
      console.warn("photo fetch failed, will retry later:", err.message);
    } finally {
      this.inflight.delete(key);
    }
  }

  // True while Giphy is configured but failing, for example when the key has
  // hit its hourly limit. Photos fill in until it recovers.
  _giphyTroubled() {
    return Date.now() - this.giphyTroubleAt < RETRY_COOLDOWN_MS;
  }

  // Starts fetches for the given moods without blocking.
  warm(moods) {
    if (this.giphyAvailable !== false) {
      for (const m of moods) {
        if ((this.moodQueues[m] || []).length < MIN_POOL) this._fetchGiphy(m);
      }
    }
    const photosNeeded = this.giphyAvailable !== true || this._giphyTroubled();
    if (photosNeeded && this.generalPool.length < MIN_POOL) this._fetchGeneral();
  }

  // Next entry for the strongest mood that has one. Giphy results come out in
  // the order Giphy ranked them. Returns null while everything is loading.
  pick(moods, excludeKey = null) {
    this.warm(moods);

    if (this.giphyAvailable !== false) {
      for (const m of moods) {
        const queue = this.moodQueues[m] || [];
        const idx = queue.findIndex((e) => e.key !== excludeKey);
        if (idx >= 0) {
          const [chosen] = queue.splice(idx, 1);
          this.seen.add(chosen.key);
          return chosen;
        }
      }
      // While Giphy works, wait for its next batch instead of mixing sources.
      if (this.giphyAvailable === true && !this._giphyTroubled()) return null;
    }

    let candidates = this.generalPool.filter((e) => e.key !== excludeKey);
    if (!candidates.length) candidates = this.generalPool;
    if (!candidates.length) return null;
    const chosen = candidates[Math.floor(Math.random() * candidates.length)];
    this.generalPool = this.generalPool.filter((e) => e.key !== chosen.key);
    this.seen.add(chosen.key);
    return chosen;
  }
}
