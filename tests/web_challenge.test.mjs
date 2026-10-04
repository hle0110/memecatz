// Run with: node --test tests/web_challenge.test.mjs
// Exercises the pure daily challenge logic in web/challenge.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  dailyPrompts, dailyNumber, parsePromptList, isDateString, previousDate, RoundTimer,
  scoreRun, compareResults, shareText, recordResult, currentStreak, bestRoundIndex, normalizeStore,
  roundSignal, faceOnlyVector, FACE_PROMPTS, GESTURE_PROMPTS, ROUND_LIMIT_MS, LAUNCH_DATE, PROMPTS,
} from "../web/challenge.js";
import { combine, topTags } from "../web/mood.js";
import { tagsFromGestures } from "../web/face.js";

const dates = (start, n) => Array.from({ length: n }, (_, i) =>
  new Date(Date.parse(`${start}T00:00:00Z`) + i * 86400000).toISOString().slice(0, 10));

test("same date gives the same prompts", () => {
  assert.deepEqual(dailyPrompts("2026-10-04"), dailyPrompts("2026-10-04"));
  assert.deepEqual(dailyPrompts("2027-02-28"), dailyPrompts("2027-02-28"));
});

test("different dates give different prompts", () => {
  assert.notDeepEqual(dailyPrompts("2026-10-04"), dailyPrompts("2026-10-05"));
  const days = dates("2026-10-04", 60);
  const distinct = new Set(days.map((d) => dailyPrompts(d).join(",")));
  assert.ok(distinct.size >= 58, `only ${distinct.size} distinct lists in 60 days`);
  for (let i = 1; i < days.length; i++) {
    assert.notDeepEqual(dailyPrompts(days[i]), dailyPrompts(days[i - 1]), `${days[i]} repeats the day before`);
  }
});

test("every day mixes 3 face and 2 gesture prompts with no repeats", () => {
  const seen = new Set();
  for (const d of dates("2026-01-01", 400)) {
    const p = dailyPrompts(d);
    assert.equal(p.length, 5);
    assert.equal(new Set(p).size, 5, `${d} repeats a prompt`);
    assert.equal(p.filter((x) => FACE_PROMPTS.includes(x)).length, 3, d);
    assert.equal(p.filter((x) => GESTURE_PROMPTS.includes(x)).length, 2, d);
    assert.ok(!p.includes("fist") && !p.includes("determined") && !p.includes("confused"));
    p.forEach((x) => seen.add(x));
  }
  assert.equal(seen.size, 9, "every prompt shows up over a year");
  for (const x of seen) assert.ok(PROMPTS[x], `${x} has prompt text`);
});

test("gesture and face prompts both land in every position over time", () => {
  const firstKinds = new Set(dates("2026-01-01", 100).map((d) => PROMPTS[dailyPrompts(d)[0]].kind));
  assert.deepEqual([...firstKinds].sort(), ["face", "gesture"]);
});

test("daily number counts from the launch date", () => {
  assert.equal(dailyNumber(LAUNCH_DATE), 1);
  assert.equal(dailyNumber("2026-10-05", "2026-10-04"), 2);
  assert.equal(dailyNumber("2027-10-04", "2026-10-04"), 366);
  assert.equal(dailyNumber("2028-03-01", "2028-02-28"), 3, "leap day counts");
});

test("date and prompt overrides are validated", () => {
  assert.ok(isDateString("2026-10-04"));
  assert.ok(!isDateString("2026-02-30"));
  assert.ok(!isDateString("10/04/2026"));
  assert.equal(previousDate("2026-03-01"), "2026-02-28");
  assert.deepEqual(parsePromptList("happy,approval,chill,suspicious,angry"), ["happy", "approval", "chill", "suspicious", "angry"]);
  assert.deepEqual(parsePromptList(" Happy , stop "), ["happy", "stop"]);
  assert.equal(parsePromptList("happy,fist"), null);
  assert.equal(parsePromptList("confused"), null, "confused is not in v1");
  assert.deepEqual(parsePromptList("neutral"), ["neutral"]);
  assert.equal(parsePromptList("happy,happy"), null);
  assert.equal(parsePromptList(""), null);
  assert.equal(parsePromptList("happy,sad,angry,surprise,confused,stop"), null);
});

// Feeds a timer frames every 33 ms. moodAt(t) gives the ranked face-only
// moods at time t, handsAt(t) the hand shapes, faceAt(t) whether the face is
// visible.
function run(timer, moodAt, faceAt = () => true, untilMs = 20000, handsAt = () => []) {
  for (let t = 0; t <= untilMs; t += 33) {
    const out = timer.update(t, roundSignal(timer.target, moodAt(t), handsAt(t)), faceAt(t));
    if (out) return { ...out, at: t };
  }
  return null;
}

test("hold timer passes after holding the target for 0.8 s", () => {
  const out = run(new RoundTimer("happy"), (t) => (t < 1000 ? [["neutral", 0.5]] : [["happy", 0.6], ["neutral", 0.1]]));
  assert.equal(out.passed, true);
  assert.ok(out.ms >= 1800 && out.ms < 1900, `passed at ${out.ms}`);
});

test("hold timer needs the target on top and at 0.35 or more", () => {
  const weak = run(new RoundTimer("happy"), () => [["happy", 0.34]]);
  assert.equal(weak.passed, false);
  assert.equal(weak.ms, ROUND_LIMIT_MS);
  assert.equal(weak.detected, null, "only the target was seen, so nothing else to blame");
  const second = run(new RoundTimer("happy"), () => [["surprise", 0.9], ["happy", 0.8]]);
  assert.equal(second.passed, false);
  assert.equal(second.detected, "surprise");
});

test("hold timer resets when the hold is broken", () => {
  const flicker = (t) => (Math.floor(t / 600) % 2 === 0 ? [["chill", 0.9]] : [["happy", 0.9]]);
  const out = run(new RoundTimer("chill"), flicker);
  assert.equal(out.passed, false, "0.6 s pieces never add up to a 0.8 s hold");
  assert.equal(out.detected, "happy");
});

test("hold timer pauses while the face is lost", () => {
  const timer = new RoundTimer("sad");
  const lost = (t) => !(t >= 2000 && t < 12000);
  const out = run(timer, () => [["neutral", 0.6]], lost, 30000);
  assert.equal(out.passed, false);
  assert.equal(out.ms, ROUND_LIMIT_MS);
  assert.ok(out.at > 15000, `missed at ${out.at}, the 10 s without a face should not count`);
  assert.equal(out.detected, "neutral");

  const broken = new RoundTimer("sad");
  const face = (t) => !(t >= 500 && t < 700);
  const passed = run(broken, () => [["sad", 0.5]], face);
  assert.equal(passed.passed, true);
  assert.ok(passed.at >= 1500, "losing the face restarts the hold");
});

test("a frame stall counts as at most one second", () => {
  const timer = new RoundTimer("happy");
  const calm = roundSignal("happy", [["neutral", 0.5]]);
  timer.update(0, calm, true);
  assert.equal(timer.update(60000, calm, true), null);
  assert.equal(timer.elapsed, 1000);
});

test("skip is a miss at the full round time", () => {
  const timer = new RoundTimer("stop");
  timer.update(0, roundSignal("stop", [["happy", 0.5]]), true);
  timer.update(500, roundSignal("stop", [["happy", 0.5]]), true);
  const out = timer.skip();
  assert.deepEqual(out, { prompt: "stop", passed: false, ms: ROUND_LIMIT_MS, detected: "happy", skipped: true });
  assert.equal(timer.update(900, roundSignal("stop", [], ["open_palm"]), true), out, "nothing changes after the round ends");
});

// The face-only ranking app.js feeds to face rounds, without smoothing.
const faceRanked = (fer, au = {}) => topTags(faceOnlyVector(fer, au), 3, 0.06);

test("a fist does not pass Look angry!", () => {
  const fer = { happy: 0.1, neutral: 0.85, angry: 0.05 };
  const fullMood = topTags(combine({ ferScores: fer, gestureTags: tagsFromGestures(["fist"]) }), 3, 0.06);
  assert.equal(fullMood[0][0], "angry", "in free play a fist does read as angry");
  const out = run(new RoundTimer("angry"), () => faceRanked(fer), () => true, 20000, () => ["fist"]);
  assert.equal(out.passed, false);
  assert.equal(out.detected, "neutral");
  const real = run(new RoundTimer("angry"), () => faceRanked({ angry: 0.8, neutral: 0.2 }, { brow_furrow: 0.6 }), () => true, 20000, () => ["fist"]);
  assert.equal(real.passed, true, "an angry face still passes with a fist up");
});

test("a smile does not pass a gesture round", () => {
  const smile = () => faceRanked({ happy: 0.95, neutral: 0.05 }, { smile: 1 });
  for (const prompt of GESTURE_PROMPTS) {
    const out = run(new RoundTimer(prompt), smile);
    assert.equal(out.passed, false, prompt);
    assert.equal(out.detected, "happy", prompt);
  }
  const wrongHand = run(new RoundTimer("approval"), smile, () => true, 20000, () => ["peace"]);
  assert.equal(wrongHand.passed, false);
  assert.equal(wrongHand.detected, "chill", "the wrong hand shape is what gets named");
});

test("gesture rounds pass on the hand shape held 0.8 s", () => {
  const pairs = { approval: "thumbs_up", chill: "peace", stop: "open_palm", suspicious: "pointing" };
  for (const [prompt, shape] of Object.entries(pairs)) {
    const out = run(new RoundTimer(prompt), () => faceRanked({ neutral: 0.9 }), () => true, 20000, (t) => (t < 1500 ? [] : [shape]));
    assert.equal(out.passed, true, prompt);
    assert.ok(out.ms >= 2300 && out.ms < 2400, `${prompt} passed at ${out.ms}`);
  }
  const brief = run(new RoundTimer("chill"), () => faceRanked({ neutral: 0.9 }), () => true, 20000,
    (t) => (Math.floor(t / 500) % 2 ? ["peace"] : []));
  assert.equal(brief.passed, false, "half second flashes are not a hold");
});

test("poker face passes on a neutral face only", () => {
  const calm = run(new RoundTimer("neutral"), () => faceRanked({ neutral: 0.9, happy: 0.1 }));
  assert.equal(calm.passed, true);
  const smiling = run(new RoundTimer("neutral"), () => faceRanked({ neutral: 0.5, happy: 0.5 }, { smile: 0.6 }));
  assert.equal(smiling.passed, false);
  assert.equal(smiling.detected, "happy");
  const weak = run(new RoundTimer("neutral"), () => [["neutral", 0.3]]);
  assert.equal(weak.passed, false, "needs 0.35 or more");
  const handsUp = run(new RoundTimer("neutral"), () => faceRanked({ neutral: 0.9 }), () => true, 20000, () => ["thumbs_up"]);
  assert.equal(handsUp.passed, true, "hands do not count against a poker face");
});

test("scoring counts passes, then total time with misses at 6 s", () => {
  const rounds = [
    { passed: true, ms: 1200 }, { passed: false, ms: 6000 }, { passed: true, ms: 2500 },
    { passed: false, ms: 4000 }, { passed: true, ms: 900 },
  ];
  assert.deepEqual(scoreRun(rounds), { passes: 3, total: 5, totalMs: 1200 + 6000 + 2500 + 6000 + 900 });
  const a = { passes: 4, totalMs: 20000 };
  const b = { passes: 3, totalMs: 9000 };
  const c = { passes: 4, totalMs: 15000 };
  assert.ok(compareResults(a, b) < 0, "more passes wins even if slower");
  assert.ok(compareResults(c, a) < 0, "same passes, less time wins");
  assert.equal(compareResults(a, { ...a }), 0);
  assert.deepEqual([a, b, c].sort(compareResults), [c, a, b]);
});

test("share text format", () => {
  const rounds = [{ passed: true }, { passed: false }, { passed: true }, { passed: true }, { passed: true }];
  const text = shareText({ number: 12, passes: 4, total: 5, totalMs: 17430, rounds }, "https://example.test/");
  assert.equal(text, "MemeCatz Daily #12 4/5 · 17.4 s\n😺⬛😺😺😺\nhttps://example.test/");
  const practice = shareText({ number: 3, passes: 0, total: 5, totalMs: 30000, rounds: rounds.map(() => ({ passed: false })), practice: true }, "u");
  assert.equal(practice, "MemeCatz Daily #3 0/5 · 30.0 s (practice)\n⬛⬛⬛⬛⬛\nu");
});

test("first finished run is official, replays are practice, best and streak update", () => {
  const r = (date, passes, totalMs) => ({ date, number: dailyNumber(date), passes, total: 5, totalMs });
  let { store, official } = recordResult(null, r("2026-10-04", 3, 20000));
  assert.equal(official, true);
  assert.equal(store.streak.count, 1);

  ({ store, official } = recordResult(store, r("2026-10-04", 5, 8000)));
  assert.equal(official, false, "a second run the same day is practice");
  assert.equal(store.official.passes, 3);
  assert.equal(store.best.passes, 3);

  ({ store, official } = recordResult(store, r("2026-10-05", 4, 25000)));
  assert.equal(store.streak.count, 2);
  assert.equal(store.best.passes, 4);
  ({ store } = recordResult(store, r("2026-10-06", 2, 9000)));
  assert.equal(store.streak.count, 3);
  assert.equal(store.best.passes, 4, "a worse day keeps the best");
  assert.equal(currentStreak(store, "2026-10-06"), 3);
  assert.equal(currentStreak(store, "2026-10-07"), 3, "still alive the next day");
  assert.equal(currentStreak(store, "2026-10-08"), 0, "a missed day ends it");

  ({ store } = recordResult(store, r("2026-10-09", 1, 29000)));
  assert.equal(store.streak.count, 1, "a gap restarts the streak");

  ({ store, official } = recordResult(store, r("2026-10-10", 5, 5000), { practice: true }));
  assert.equal(official, false, "test prompt runs never count");
  assert.equal(store.official.date, "2026-10-09");
});

test("damaged saved data is ignored", () => {
  assert.deepEqual(normalizeStore("nope"), normalizeStore(null));
  const s = normalizeStore({ official: { date: "yesterday" }, best: { passes: "5" }, streak: { count: -2, lastDate: "x" } });
  assert.deepEqual(s, { official: null, best: null, streak: { count: 0, lastDate: null } });
});

test("best round for the share image prefers fast passes with a usable image", () => {
  const rounds = [
    { passed: false, ms: 6000, cors: true }, { passed: true, ms: 3000, cors: true },
    { passed: true, ms: 1000, cors: false }, { passed: true, ms: 2000, cors: true },
  ];
  assert.equal(bestRoundIndex(rounds, (r) => r.cors), 3);
  assert.equal(bestRoundIndex(rounds), 2);
  assert.equal(bestRoundIndex([{ passed: false, ms: 6000, cors: false }], (r) => r.cors), -1);
});
