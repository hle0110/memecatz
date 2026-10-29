// Run with: node --test tests/web_reactions.test.mjs
// Exercises web/reactions.js with a fake fetch, no network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import { ReactionSource, DOG_BREEDS, MOOD_QUERIES, isLibraryItem, libraryEntries, nextLibraryItem } from "../web/reactions.js";

const tick = () => new Promise((r) => setTimeout(r, 0));
const ids = ["a", "b", "c", "d", "e", "f", "g", "h"];
const giphyBody = { data: ids.map((id) => ({ id, images: { fixed_height: { url: `https://media.giphy.com/${id}.gif`, webp: `https://media.giphy.com/${id}.webp` } } })) };
const reply = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });

function fakeFetch({ key = "K", giphyStatus = 200 } = {}) {
  const calls = [];
  const fn = async (url) => {
    calls.push(String(url));
    if (url === "/api/config") return key === null ? reply(404, {}) : reply(200, { giphyKey: key });
    if (String(url).startsWith("https://api.giphy.com/")) return reply(giphyStatus, giphyStatus === 200 ? giphyBody : {});
    if (String(url).startsWith("https://api.thecatapi.com/")) return reply(200, [{ id: "c1", url: "https://cdn2.thecatapi.com/c1.jpg" }, { id: "c2", url: "https://cdn2.thecatapi.com/c2.jpg" }]);
    if (String(url).startsWith("https://dog.ceo/")) return reply(200, { status: "success", message: ["https://images.dog.ceo/breeds/pug/1.jpg"] });
    throw new Error("unexpected " + url);
  };
  return { fn, calls };
}

test("giphy results come out in giphy's order, as webp, with attribution", async () => {
  const f = fakeFetch();
  const rs = new ReactionSource("cat", f.fn);
  rs.warm(["happy"]);
  for (let i = 0; i < 5; i++) await tick();
  const picks = [rs.pick(["happy"]), rs.pick(["happy"]), rs.pick(["happy"])];
  assert.deepEqual(picks.map((p) => p.key), ["giphy:a", "giphy:b", "giphy:c"]);
  assert.equal(picks[0].url, "https://media.giphy.com/a.webp");
  assert.equal(picks[0].attribution, "Powered By GIPHY");
  const search = f.calls.find((u) => u.startsWith("https://api.giphy.com/"));
  assert.match(search, /q=cat\+smiling/);
  assert.match(search, /limit=12/);
  assert.match(search, /rating=g/);
  assert.ok(!f.calls.some((u) => u.includes("/api/giphy")), "never goes through a proxy");
});

test("no key configured falls back to real photos", async () => {
  const f = fakeFetch({ key: null });
  const rs = new ReactionSource("cat", f.fn);
  rs.warm(["happy"]);
  for (let i = 0; i < 5; i++) await tick();
  const p = rs.pick(["happy"]);
  assert.equal(p.source, "cat_api");
  assert.equal(p.cors, false);
  assert.ok(!f.calls.some((u) => u.startsWith("https://api.giphy.com/")));
});

test("rejected key stops giphy and uses photos", async () => {
  const f = fakeFetch({ giphyStatus: 403 });
  const rs = new ReactionSource("dog", f.fn);
  rs.warm(["happy"]);
  for (let i = 0; i < 5; i++) await tick();
  assert.equal(rs.giphyAvailable, false);
  rs.warm(["happy"]);
  for (let i = 0; i < 5; i++) await tick();
  const p = rs.pick(["happy"]);
  assert.equal(p.source, "dog_api");
  const dogCall = f.calls.find((u) => u.startsWith("https://dog.ceo/"));
  assert.ok(DOG_BREEDS.some((b) => dogCall === `https://dog.ceo/api/breed/${b}/images/random/8`), dogCall);
});

test("rate limited giphy lets photos fill in", async () => {
  const f = fakeFetch({ giphyStatus: 429 });
  const rs = new ReactionSource("cat", f.fn);
  rs.warm(["happy"]);
  for (let i = 0; i < 5; i++) await tick();
  rs.warm(["happy"]);
  for (let i = 0; i < 5; i++) await tick();
  const p = rs.pick(["happy"]);
  assert.ok(p, "shows something instead of waiting forever");
  assert.equal(p.source, "cat_api");
});

test("exclude key skips the reaction already on screen", async () => {
  const f = fakeFetch();
  const rs = new ReactionSource("cat", f.fn);
  rs.warm(["sad"]);
  for (let i = 0; i < 5; i++) await tick();
  const p = rs.pick(["sad"], "giphy:a");
  assert.equal(p.key, "giphy:b");
});

test("a network error on /api/config falls back to photos without disabling giphy for good", async () => {
  let configCalls = 0;
  const base = fakeFetch();
  const fn = async (url) => {
    if (url === "/api/config") {
      configCalls += 1;
      throw new Error("offline");
    }
    return base.fn(url);
  };
  const rs = new ReactionSource("cat", fn);
  rs.warm(["happy"]);
  for (let i = 0; i < 5; i++) await tick();
  assert.notEqual(rs.giphyAvailable, false, "a blip is not a final no");
  const p = rs.pick(["happy"]);
  assert.equal(p.source, "cat_api");
  assert.equal(configCalls, 1);
});

test("each mood rotates through its phrases, each limited to its depth", async () => {
  const f = fakeFetch();
  const rs = new ReactionSource("cat", f.fn);
  const phrases = MOOD_QUERIES.cat.annoyed;
  assert.ok(phrases.length >= 2);
  for (let round = 0; round < phrases.length + 1; round++) {
    rs.moodQueues.annoyed = [];
    rs.warm(["annoyed"]);
    for (let i = 0; i < 5; i++) await tick();
  }
  const searches = f.calls.filter((u) => u.startsWith("https://api.giphy.com/")).map((u) => new URL(u));
  const expected = [...phrases, phrases[0]];
  assert.deepEqual(searches.map((u) => [u.searchParams.get("q"), Number(u.searchParams.get("limit"))]),
                   expected.map(([q, d]) => [q, d]));
  assert.ok(searches.every((u) => u.searchParams.get("offset") === "0"));
});

test("dog mode uses the dog phrases", async () => {
  const f = fakeFetch();
  const rs = new ReactionSource("dog", f.fn);
  rs.warm(["confused"]);
  for (let i = 0; i < 5; i++) await tick();
  const search = new URL(f.calls.find((u) => u.startsWith("https://api.giphy.com/")));
  assert.equal(search.searchParams.get("q"), MOOD_QUERIES.dog.confused[0][0]);
});

const clip = (file, extra = {}) => ({ file, type: file.endsWith(".mp4") ? "mp4" : "webp", credit: "Mochi, filmed by the owner", permission: "owner", ...extra });

test("pet library items come first for their mood, take turns, and skip Giphy", async () => {
  const f = fakeFetch();
  const rs = new ReactionSource("dog", f.fn);
  rs.setLibrary({ dog: { stop: [clip("dog/stop-1.webp"), clip("dog/stop-2.mp4", { permission: "with permission" })] } });
  const picks = [rs.pick(["stop"]), rs.pick(["stop"]), rs.pick(["stop"])];
  assert.deepEqual(picks.map((p) => p.key), ["pet:dog/stop-1.webp", "pet:dog/stop-2.mp4", "pet:dog/stop-1.webp"]);
  assert.equal(picks[0].url, "/pets/dog/stop-1.webp");
  assert.equal(picks[1].type, "mp4");
  assert.equal(picks[0].attribution, "Pet clip: Mochi, filmed by the owner");
  assert.equal(picks[0].source, "pet_library");
  for (let i = 0; i < 5; i++) await tick();
  assert.ok(!f.calls.some((u) => u.startsWith("https://api.giphy.com/")), "a mood with library items never searches Giphy");
  assert.equal(rs.pick(["stop"], "pet:dog/stop-2.mp4").key, "pet:dog/stop-1.webp", "skips the one on screen");
});

test("a mood without library items falls back to Giphy as before", async () => {
  const f = fakeFetch();
  const rs = new ReactionSource("dog", f.fn);
  rs.setLibrary({ dog: { stop: [], sad: [clip("../secret.webp"), { file: "dog/x.webp", type: "webp", credit: " ", permission: "owner" }] } });
  rs.warm(["stop", "sad"]);
  for (let i = 0; i < 5; i++) await tick();
  assert.equal(rs.pick(["stop"]).key, "giphy:a");
  assert.equal(rs.pick(["sad"]).key, "giphy:a", "only broken items is the same as none");
  const queries = f.calls.filter((u) => u.startsWith("https://api.giphy.com/")).map((u) => new URL(u).searchParams.get("q"));
  assert.deepEqual(queries.sort(), ["dog high five", "sad dog"]);
});

test("library items must be well formed", () => {
  assert.ok(isLibraryItem(clip("cat/approval.webp")));
  assert.ok(isLibraryItem(clip("approval.mp4")));
  for (const bad of [
    clip("cat/a.webp", { type: "gif" }), clip("cat/a.mp4", { type: "webp" }), clip("cat/a.webp", { permission: "found online" }),
    clip("cat/a.webp", { credit: "" }), clip("cat/a.webp", { credit: 5 }), clip("/etc/a.webp"), clip("cat/../a.webp"),
    clip("cat//a.webp"), clip("https://example.com/a.webp"), clip("cat/a b.webp"), null, [], "cat/a.webp",
  ]) assert.equal(isLibraryItem(bad), false, JSON.stringify(bad));
  assert.deepEqual(libraryEntries({ cat: { happy: [clip("h.webp", { credit: "  Mochi \n" })] } }, "happy", "cat")[0].credit, "Mochi");
  assert.deepEqual(libraryEntries(null, "happy"), []);
  assert.deepEqual(libraryEntries({ cat: [] }, "happy"), []);
  assert.deepEqual(nextLibraryItem([], 3), [null, 3]);
});

test("a licensed clip needs its source page and license", () => {
  const lic = { source: "https://www.pexels.com/video/dog-giving-a-high-five-123/", license: "Pexels License" };
  const item = clip("dog/stop-1.webp", { permission: "licensed", credit: "Jane Doe on Pexels", ...lic });
  assert.ok(isLibraryItem(item));
  const [entry] = libraryEntries({ dog: { stop: [{ ...item, license: " Pexels License\n" }] } }, "stop", "dog");
  assert.equal(entry.source, lic.source);
  assert.equal(entry.license, "Pexels License");
  for (const bad of [
    { source: undefined }, { source: "http://www.pexels.com/video/1/" }, { source: "https://x.test/a b" },
    { source: "javascript:alert(1)" }, { license: "" }, { license: " \t" }, { license: undefined },
  ]) assert.equal(isLibraryItem({ ...item, ...bad }), false, JSON.stringify(bad));
  const own = libraryEntries({ dog: { stop: [clip("dog/s.webp", lic)] } }, "stop", "dog")[0];
  assert.equal(own.source, null, "source and license only matter for licensed clips");
  assert.equal(own.license, null);
});

test("the checked-in library parses, and every item is valid, present and under 500 KB", () => {
  const lib = JSON.parse(readFileSync(new URL("../web/pets/library.json", import.meta.url), "utf8"));
  assert.deepEqual(Object.keys(lib.cat).sort(), ["approval", "stop"]);
  assert.deepEqual(Object.keys(lib.dog).sort(), ["approval", "sad", "stop", "suspicious"]);
  for (const animal of ["cat", "dog"]) {
    for (const [mood, items] of Object.entries(lib[animal])) {
      assert.ok(Array.isArray(items), `${animal} ${mood}`);
      for (const item of items) {
        assert.ok(isLibraryItem(item), `${animal} ${mood}: ${JSON.stringify(item)} is malformed`);
        const size = statSync(new URL(`../web/pets/${item.file}`, import.meta.url)).size;
        assert.ok(size < 500 * 1024, `${item.file} is ${Math.round(size / 1024)} KB, keep it under 500 KB`);
      }
    }
  }
});
