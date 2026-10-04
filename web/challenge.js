// Daily challenge: five prompts that are the same for everyone on a given UTC
// date. Everything here is pure (no page, no clock, no storage) so it can be
// tested in Node. web/app.js drives it with the live signals and the real time.

import { combine } from "./mood.js";

// Day #1 is the day the challenge first went live.
export const LAUNCH_DATE = "2026-10-04";

export const ROUND_LIMIT_MS = 6000;
export const HOLD_MS = 800;
export const PASS_SCORE = 0.35;
// A stalled frame (a busy device, a hidden tab) counts as at most this much
// time, so one long gap cannot use up a round.
export const MAX_STEP_MS = 1000;

export const FACE_PROMPTS = ["happy", "surprise", "angry", "sad", "neutral"];
export const GESTURE_PROMPTS = ["approval", "chill", "stop", "suspicious"];
// Gesture rounds check the classified hand shape itself, not the mood.
export const GESTURE_FOR_PROMPT = { approval: "thumbs_up", chill: "peace", stop: "open_palm", suspicious: "pointing" };
// What a shown hand shape means, for "you looked ... instead".
export const GESTURE_MOOD = {
  thumbs_up: "approval", thumbs_down: "disapproval", peace: "chill", open_palm: "stop", pointing: "suspicious", fist: "angry",
};
export const FACE_COUNT = 3;
export const GESTURE_COUNT = 2;

export const PROMPTS = {
  happy: { kind: "face", text: "Big smile!", hint: "show some teeth" },
  surprise: { kind: "face", text: "Look shocked!", hint: "eyebrows up, mouth open" },
  angry: { kind: "face", text: "Look angry!", hint: "frown those brows" },
  sad: { kind: "face", text: "Look sad!", hint: "corners of the mouth down" },
  neutral: { kind: "face", text: "Poker face!", hint: "no expression at all" },
  approval: { kind: "gesture", text: "Thumbs up!", hint: "thumb high, fingers curled" },
  chill: { kind: "gesture", text: "Peace sign!", hint: "two fingers up" },
  stop: { kind: "gesture", text: "Hand up, stop!", hint: "open palm, all five fingers" },
  suspicious: { kind: "gesture", text: "Point a finger!", hint: "just the index finger" },
};

const DAY_MS = 86400000;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export function utcDateString(date) {
  return date.toISOString().slice(0, 10);
}

export function isDateString(text) {
  if (typeof text !== "string" || !DATE_RE.test(text)) return false;
  const d = new Date(`${text}T00:00:00Z`);
  return !Number.isNaN(d.getTime()) && utcDateString(d) === text;
}

function dayIndex(dateStr) {
  return Math.round(Date.parse(`${dateStr}T00:00:00Z`) / DAY_MS);
}

export function dailyNumber(dateStr, launch = LAUNCH_DATE) {
  return dayIndex(dateStr) - dayIndex(launch) + 1;
}

export function previousDate(dateStr) {
  return utcDateString(new Date(Date.parse(`${dateStr}T00:00:00Z`) - DAY_MS));
}

// FNV-1a, 32 bit. Turns the date string into a seed.
export function hashString(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

// mulberry32: small, fast, and the same in every browser.
export function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle(items, rand) {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// Three face prompts and two gesture prompts, no repeats, in an order that
// depends only on the date.
export function dailyPrompts(dateStr) {
  const rand = seededRandom(hashString(`memecatz-daily:${dateStr}`));
  const faces = shuffle(FACE_PROMPTS, rand).slice(0, FACE_COUNT);
  const gestures = shuffle(GESTURE_PROMPTS, rand).slice(0, GESTURE_COUNT);
  return shuffle([...faces, ...gestures], rand);
}

// "?prompts=happy,approval" for testing. Unknown names or repeats make the
// whole override invalid, so a typo never silently changes the run.
export function parsePromptList(text) {
  if (typeof text !== "string" || !text.trim()) return null;
  const list = text.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  if (!list.length || list.length > 5) return null;
  if (list.some((p) => !PROMPTS[p])) return null;
  if (new Set(list).size !== list.length) return null;
  return list;
}

// The mood from the face alone: the emotion model and the expression
// strengths, with no hand signals, so a fist cannot pass "Look angry!".
export function faceOnlyVector(ferScores, auTags) {
  return combine({ ferScores, auTags });
}

// Whether this frame meets the prompt, and what was shown instead.
// faceRanked: ranked smoothed face-only moods, [[tag, score], ...].
// gestures: the hand shapes classified in the latest hand check.
export function roundSignal(prompt, faceRanked, gestures = []) {
  const top = faceRanked && faceRanked.length ? faceRanked[0] : null;
  const wanted = GESTURE_FOR_PROMPT[prompt];
  if (wanted) {
    if (gestures.includes(wanted)) return { hit: true, seen: prompt };
    const other = gestures.find((g) => GESTURE_MOOD[g]);
    return { hit: false, seen: other ? GESTURE_MOOD[other] : top ? top[0] : null };
  }
  return { hit: !!top && top[0] === prompt && top[1] >= PASS_SCORE, seen: top ? top[0] : null };
}

// One round. Call update() on every camera frame with the time in ms, the
// frame's roundSignal() and whether a face is in view. The round clock only
// runs while the face is visible. Passing needs a hit for HOLD_MS without a
// break.
export class RoundTimer {
  constructor(target, { limitMs = ROUND_LIMIT_MS, holdMs = HOLD_MS, maxStepMs = MAX_STEP_MS } = {}) {
    this.target = target;
    this.limitMs = limitMs;
    this.holdMs = holdMs;
    this.maxStepMs = maxStepMs;
    this.elapsed = 0;
    this.held = 0;
    this.holding = false;
    this.paused = false;
    this.last = null;
    this.topTime = {};
    this.outcome = null;
  }

  update(now, signal, faceVisible) {
    if (this.outcome) return this.outcome;
    const dt = this.last === null ? 0 : Math.min(Math.max(0, now - this.last), this.maxStepMs);
    this.last = now;
    if (!faceVisible) {
      this.paused = true;
      this.holding = false;
      this.held = 0;
      return null;
    }
    this.paused = false;
    const step = Math.min(dt, this.limitMs - this.elapsed);
    this.elapsed += step;

    const seen = signal ? signal.seen : null;
    if (seen) this.topTime[seen] = (this.topTime[seen] || 0) + step;
    const hit = !!(signal && signal.hit);
    if (hit && this.holding) this.held += step;
    else if (hit) { this.holding = true; this.held = 0; }
    else { this.holding = false; this.held = 0; }

    if (hit && this.held >= this.holdMs) {
      this.outcome = { prompt: this.target, passed: true, ms: this.elapsed, detected: this.target, skipped: false };
    } else if (this.elapsed >= this.limitMs) {
      this.outcome = { prompt: this.target, passed: false, ms: this.limitMs, detected: this.mostSeen(), skipped: false };
    }
    return this.outcome;
  }

  skip() {
    if (!this.outcome) {
      this.outcome = { prompt: this.target, passed: false, ms: this.limitMs, detected: this.mostSeen(), skipped: true };
    }
    return this.outcome;
  }

  // What was shown the longest this round, other than the target.
  // null when only the target (too weak) or nothing was seen.
  mostSeen() {
    let best = null;
    for (const [tag, ms] of Object.entries(this.topTime)) {
      if (tag === this.target || ms <= 0) continue;
      if (!best || ms > best[1]) best = [tag, ms];
    }
    return best ? best[0] : null;
  }

  get remainingMs() { return Math.max(0, this.limitMs - this.elapsed); }
  get holdProgress() { return Math.min(1, this.held / this.holdMs); }
}

export function scoreRun(rounds) {
  let passes = 0;
  let totalMs = 0;
  for (const r of rounds) {
    if (r.passed) { passes += 1; totalMs += r.ms; }
    else totalMs += ROUND_LIMIT_MS;
  }
  return { passes, total: rounds.length, totalMs };
}

// Negative when a is the better result: more passes first, then less time.
export function compareResults(a, b) {
  return (b.passes - a.passes) || (a.totalMs - b.totalMs);
}

export function formatSeconds(ms) {
  return (ms / 1000).toFixed(1);
}

export function emojiRow(rounds) {
  return rounds.map((r) => (r.passed ? "😺" : "⬛")).join("");
}

export function shareText(result, url) {
  const head = `MemeCatz Daily #${result.number} ${result.passes}/${result.total} · ${formatSeconds(result.totalMs)} s`;
  return [head + (result.practice ? " (practice)" : ""), emojiRow(result.rounds), url].join("\n");
}

// The round to put on the share image: passes before misses, faster first,
// and only rounds whose reaction image may be drawn into a canvas.
export function bestRoundIndex(rounds, usable = () => true) {
  let best = -1;
  rounds.forEach((r, i) => {
    if (!usable(r, i)) return;
    if (best < 0) { best = i; return; }
    const b = rounds[best];
    if ((r.passed && !b.passed) || (r.passed === b.passed && r.ms < b.ms)) best = i;
  });
  return best;
}

// Saved state: { official, best, streak: { count, lastDate } }. The first
// finished run of a date is the official one; later runs that day are
// practice and change nothing.
export function emptyStore() {
  return { official: null, best: null, streak: { count: 0, lastDate: null } };
}

export function normalizeStore(raw) {
  const store = emptyStore();
  if (!raw || typeof raw !== "object") return store;
  if (raw.official && isDateString(raw.official.date)) store.official = raw.official;
  if (raw.best && typeof raw.best.passes === "number" && typeof raw.best.totalMs === "number") store.best = raw.best;
  if (raw.streak && Number.isInteger(raw.streak.count) && raw.streak.count >= 0 &&
      (raw.streak.lastDate === null || isDateString(raw.streak.lastDate))) {
    store.streak = { count: raw.streak.count, lastDate: raw.streak.lastDate };
  }
  return store;
}

export function isOfficialDone(store, dateStr) {
  return !!(store.official && store.official.date === dateStr);
}

// result: { date, number, passes, total, totalMs, marks }. Returns the new
// store and whether this run became the day's official result.
export function recordResult(rawStore, result, { practice = false } = {}) {
  const store = normalizeStore(rawStore);
  if (practice || isOfficialDone(store, result.date)) return { store, official: false };
  const { lastDate, count } = store.streak;
  const streakCount = lastDate === previousDate(result.date) ? count + 1 : 1;
  return {
    store: {
      official: result,
      best: !store.best || compareResults(result, store.best) < 0 ? result : store.best,
      streak: { count: streakCount, lastDate: result.date },
    },
    official: true,
  };
}

// The streak as it stands today: still alive if the last official run was
// today or yesterday.
export function currentStreak(rawStore, dateStr) {
  const { count, lastDate } = normalizeStore(rawStore).streak;
  return lastDate === dateStr || lastDate === previousDate(dateStr) ? count : 0;
}
