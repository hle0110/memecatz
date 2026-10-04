// Run with: node --test tests/web_reactions.test.mjs
// Exercises web/reactions.js with a fake fetch, no network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { ReactionSource, DOG_BREEDS, MOOD_QUERIES } from "../web/reactions.js";

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
