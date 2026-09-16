// Real cat and dog photos from two public APIs. No key, no secrets.
// Both calls are plain GETs with no custom headers, which avoids a CORS
// preflight. If a fetch fails the app shows an honest waiting state rather
// than substituting a fake image.

const CAT_API_SEARCH_URL = "https://api.thecatapi.com/v1/images/search";
const DOG_API_URL = "https://dog.ceo/api/breeds/image/random";
const BATCH_SIZE = 6;
const RETRY_COOLDOWN_MS = 30000;

export class ReactionSource {
  constructor(animal = "cat") {
    this.animal = animal === "dog" ? "dog" : "cat";
    this.pool = [];
    this.lastFailureAt = 0;
    this.fetching = false;
  }

  setAnimal(animal) {
    const next = animal === "dog" ? "dog" : "cat";
    if (next !== this.animal) {
      this.animal = next;
      this.pool = [];
      this.lastFailureAt = 0;
    }
  }

  describeSource() {
    return this.animal === "dog"
      ? "real dog photos via Dog CEO API"
      : "real cat photos via The Cat API";
  }

  async _fetchCats() {
    const res = await fetch(`${CAT_API_SEARCH_URL}?limit=${BATCH_SIZE}`);
    if (!res.ok) throw new Error(`cat api ${res.status}`);
    const payload = await res.json();
    if (!Array.isArray(payload)) throw new Error("unexpected cat api payload");
    return payload
      .filter((item) => item && item.id && item.url)
      .map((item) => ({ key: `catapi:${item.id}`, url: item.url, name: "real cat photo" }));
  }

  async _fetchDogs() {
    const results = [];
    for (let i = 0; i < BATCH_SIZE; i++) {
      const res = await fetch(DOG_API_URL);
      if (!res.ok) throw new Error(`dog api ${res.status}`);
      const payload = await res.json();
      if (payload.status !== "success" || !payload.message) throw new Error("unexpected dog api payload");
      const url = payload.message;
      results.push({ key: `dogapi:${url}`, url, name: "real dog photo" });
    }
    return results;
  }

  async refill() {
    if (this.fetching) return;
    if (Date.now() - this.lastFailureAt < RETRY_COOLDOWN_MS) return;
    this.fetching = true;
    try {
      const entries = this.animal === "dog" ? await this._fetchDogs() : await this._fetchCats();
      if (entries.length) this.pool.push(...entries);
      else this.lastFailureAt = Date.now();
    } catch (err) {
      this.lastFailureAt = Date.now();
      console.warn("reaction fetch failed, will retry:", err.message);
    } finally {
      this.fetching = false;
    }
  }

  // Returns an entry, or null when nothing real is available yet.
  pick(excludeKey = null) {
    if (this.pool.length <= 2) this.refill();
    if (!this.pool.length) return null;

    let candidates = this.pool.filter((e) => e.key !== excludeKey);
    if (!candidates.length) candidates = this.pool;
    const chosen = candidates[Math.floor(Math.random() * candidates.length)];
    this.pool = this.pool.filter((e) => e.key !== chosen.key);
    return chosen;
  }
}
