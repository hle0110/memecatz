// Real reactions from three kinds of source, in order of preference:
//
//   0. The own-pet library, mood matched: clips of the owner's own pets, or of
//      pets filmed with permission, served from this site under /pets/. A
//      mood with library items never searches Giphy. See docs/PET_LIBRARY.md.
//   1. Giphy, mood matched, for every mood the library does not cover. The page asks this site's /api/config for the
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

// Giphy search phrases per mood, the same as MOOD_QUERIES in reactions.py.
// Each phrase was chosen by looking at what Giphy actually returns for it:
// real animals showing that mood, not drawings or people. The number is how
// many of its top results to use. Past that point a phrase's results drift
// off-mood or to cartoons, so the app moves on to the next phrase instead of
// digging deeper.
// Checked by eye in October 2026. Giphy's results change over time, so it is
// worth looking at them again every few months.
export const MOOD_QUERIES = {
  cat: {
    happy: [["cat smiling", 12], ["happy cat", 7]],
    sad: [["crying cat", 11], ["sad cat", 11]],
    angry: [["angry cat", 12]],
    surprise: [["shocked cat", 12], ["surprised cat", 12]],
    fear: [["scared cat", 12]],
    disgust: [["disgusted cat", 7]],
    neutral: [["cat stare", 11]],
    smug: [["cat smirk", 10]],
    confused: [["cat huh", 12], ["confused cat", 5]],
    mischief: [["cat knocking things off", 12], ["sneaky cat", 9]],
    annoyed: [["annoyed cat", 12], ["unimpressed cat", 12], ["grumpy cat", 12]],
    approval: [["cat thumbs up", 3], ["cat smiling", 12]],
    disapproval: [["cat side eye", 9]],
    chill: [["relaxed cat", 12]],
    suspicious: [["suspicious cat", 5], ["cat side eye", 9]],
    triumph: [["cat winning", 3], ["cat smiling", 12]],
    anxious: [["scared cat", 12]],
    bored: [["bored cat", 12], ["cat yawning", 11]],
    mocking: [["cat laughing", 10], ["sassy cat", 12]],
    stop: [["cat high five", 3], ["cat no", 4], ["cat says no", 3]],
    determined: [["serious cat", 12], ["cat butt wiggle", 10]],
    focused: [["cat butt wiggle", 10], ["serious cat", 12]],
  },
  dog: {
    happy: [["dog smiling", 12], ["happy dog", 5]],
    sad: [["sad dog", 5], ["dog crying", 2], ["depressed dog", 1]],
    angry: [["angry dog", 12]],
    surprise: [["shocked dog", 10], ["surprised dog", 9]],
    fear: [["scared dog", 9]],
    disgust: [["disgusted dog", 7]],
    neutral: [["dog stare", 12], ["dog staring", 12]],
    smug: [["smug dog", 6]],
    confused: [["dog head tilt", 12], ["confused dog", 12]],
    mischief: [["dog zoomies", 10], ["guilty dog", 2]],
    annoyed: [["dog side eye", 5], ["unimpressed dog", 3], ["dog eye roll", 2]],
    approval: [["dog yes", 4], ["dog thumbs up", 2], ["dog smiling", 12]],
    disapproval: [["dog side eye", 5], ["dog judging", 6]],
    chill: [["lazy dog", 10], ["sleepy dog", 3]],
    suspicious: [["suspicious dog", 2], ["dog side eye", 5]],
    triumph: [["excited dog", 12], ["dog winning", 3]],
    anxious: [["nervous dog", 5], ["scared dog", 9]],
    bored: [["bored dog", 12], ["dog yawning", 6]],
    mocking: [["dog grin", 5], ["dog smiling", 12]],
    stop: [["dog high five", 8]],
    determined: [["serious dog", 12]],
    focused: [["serious dog", 12], ["dog staring", 12]],
  },
};

// Phrases for a mood, with a plain fallback for any mood not in the table.
export function queriesForMood(moodTag, animal = "cat") {
  const table = MOOD_QUERIES[animal === "dog" ? "dog" : "cat"];
  return table[moodTag] || [[`${moodTag.replace(/_/g, " ")} ${animal}`, 8]];
}

// The own-pet library. web/pets/library.json is shared with the desktop app,
// which reads the same file: { cat: { mood: [item, ...] }, dog: { ... } },
// where an item is { file, type, credit, permission } and file is relative to
// web/pets/.
export const PET_LIBRARY_URL = "/pets/library.json";
export const PET_LIBRARY_DIR = "/pets/";
export const PET_TYPES = ["webp", "mp4"];
export const PET_PERMISSIONS = ["owner", "with permission"];

// Only ASCII spaces, tabs and line breaks, so the desktop's strip() matches.
const trimCredit = (text) => text.replace(/^[ \t\r\n]+|[ \t\r\n]+$/g, "");

// A plain relative path inside web/pets/ that ends in the item's type.
export function isLibraryItem(item) {
  if (!item || typeof item !== "object") return false;
  const { file, type, credit, permission } = item;
  if (!PET_TYPES.includes(type) || !PET_PERMISSIONS.includes(permission)) return false;
  if (typeof credit !== "string" || !trimCredit(credit)) return false;
  if (typeof file !== "string" || !/^[A-Za-z0-9_][A-Za-z0-9_.\/-]*$/.test(file)) return false;
  if (file.split("/").some((part) => !part || part === "." || part === "..")) return false;
  return file.toLowerCase().endsWith(`.${type}`);
}

// The usable library items for a mood, in file order. A malformed item is
// left out, so a typo in the file never shows a broken reaction. An empty
// list means the mood uses Giphy.
export function libraryEntries(library, mood, animal = "cat") {
  const table = library && typeof library === "object" ? library[animal === "dog" ? "dog" : "cat"] : null;
  const items = table && typeof table === "object" ? table[mood] : null;
  if (!Array.isArray(items)) return [];
  return items.filter(isLibraryItem).map((it) => ({
    key: `pet:${it.file}`, file: it.file, type: it.type, credit: trimCredit(it.credit), permission: it.permission,
  }));
}

// The library items take turns. Returns [item, nextCursor], skipping
// excludeKey when another item exists, or [null, cursor] for no items.
export function nextLibraryItem(entries, cursor, excludeKey = null) {
  const n = entries.length;
  if (!n) return [null, cursor];
  for (let i = 0; i < n; i++) {
    const idx = (cursor + i) % n;
    if (entries[idx].key !== excludeKey) return [entries[idx], idx + 1];
  }
  const idx = cursor % n;
  return [entries[idx], idx + 1];
}

// The credit shown under a library item, the same text on desktop.
export function petCreditLine(credit) {
  return `Pet clip: ${credit}`;
}

export function libraryReaction(item, mood, animal) {
  return {
    key: item.key,
    url: PET_LIBRARY_DIR + item.file,
    type: item.type,
    name: `${mood} ${animal} reaction`,
    attribution: petCreditLine(item.credit),
    source: "pet_library",
    cors: true,
  };
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
    this.library = null;           // web/pets/library.json, once loaded
    this._reset();
  }

  _reset() {
    this.moodQueues = {};          // mood -> entries in Giphy's order
    this.phraseIndex = {};         // mood -> which of its phrases to search next
    this.generalPool = [];         // keyless photos
    this.failedAt = {};
    this.inflight = new Set();
    this.seen = new Set();
    this.libraryCursor = {};       // mood -> which library item is next
  }

  setLibrary(library) {
    this.library = library && typeof library === "object" ? library : null;
    this.libraryCursor = {};
  }

  _libraryFor(mood) {
    return libraryEntries(this.library, mood, this.animal);
  }

  _takeLibrary(mood, excludeKey) {
    const [item, cursor] = nextLibraryItem(this._libraryFor(mood), this.libraryCursor[mood] || 0, excludeKey);
    if (!item) return null;
    this.libraryCursor[mood] = cursor;
    return libraryReaction(item, mood, this.animal);
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
      const phrases = queriesForMood(mood, this.animal);
      const index = (this.phraseIndex[mood] || 0) % phrases.length;
      const [query, depth] = phrases[index];
      const url = new URL(GIPHY_SEARCH_URL);
      url.searchParams.set("api_key", this.giphyKey);
      url.searchParams.set("q", query);
      url.searchParams.set("limit", String(depth));
      url.searchParams.set("offset", "0");
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
      // Next time, the next phrase. After the last one, back to the first.
      this.phraseIndex[mood] = index + 1;
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
        if (this._libraryFor(m).length) continue;
        if ((this.moodQueues[m] || []).length < MIN_POOL) this._fetchGiphy(m);
      }
    }
    const photosNeeded = this.giphyAvailable !== true || this._giphyTroubled();
    if (photosNeeded && this.generalPool.length < MIN_POOL) this._fetchGeneral();
  }

  // Next entry for the strongest mood that has one: the pet library for that
  // mood first, else Giphy. Giphy results come out in the order Giphy ranked
  // them. Returns null while everything is loading.
  pick(moods, excludeKey = null) {
    this.warm(moods);

    for (const m of moods) {
      const own = this._takeLibrary(m, excludeKey);
      if (own) return own;
      if (this.giphyAvailable === false) continue;
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

    let candidates = this.generalPool.filter((e) => e.key !== excludeKey);
    if (!candidates.length) candidates = this.generalPool;
    if (!candidates.length) return null;
    const chosen = candidates[Math.floor(Math.random() * candidates.length)];
    this.generalPool = this.generalPool.filter((e) => e.key !== chosen.key);
    this.seen.add(chosen.key);
    return chosen;
  }
}
