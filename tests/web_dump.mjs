// Helper for test_web_parity.py. Reads a JSON request on stdin, runs the
// browser modules from web/ in Node, and prints their results as JSON.
import { readFileSync } from "node:fs";
import { combine, topTags, AU_TAG_TO_MOOD } from "../web/mood.js";
import { STATIC_CAPTIONS } from "../web/captions.js";
import { MOOD_QUERIES, DOG_BREEDS, queriesForMood } from "../web/reactions.js";
import { classifyGesturePoints, GESTURE_TO_TAGS, FEATURE_KEYS, rateAdjustedAlpha } from "../web/face.js";
import { toSquare, cropRect, grayOf } from "../web/emotion.js";

const req = JSON.parse(readFileSync(0, "utf8"));

const out = {
  captions: STATIC_CAPTIONS,
  moodQueries: MOOD_QUERIES,
  queries: Object.fromEntries(["happy", "neutral", "made_up_tag"].map((m) => [m, queriesForMood(m, "dog")])),
  dogBreeds: DOG_BREEDS,
  gestureToTags: GESTURE_TO_TAGS,
  auTagToMood: AU_TAG_TO_MOOD,
  featureKeys: FEATURE_KEYS,
  moods: req.moodCases.map((c) => {
    const v = combine(c);
    return { vector: v, top: topTags(v, 3, 0.12) };
  }),
  gestures: req.gestureCases.map((pts) => classifyGesturePoints(pts)),
  squares: req.boxes.map((b) => toSquare(b)),
  crops: req.boxes.map((b) => cropRect(toSquare(b), 640, 480)),
  grays: req.pixels.map(([r, g, b]) => grayOf(r, g, b)),
  alphaSame: rateAdjustedAlpha(0.35, 0.15, 0.15),
  alphaTwoHalves: 1 - (1 - rateAdjustedAlpha(0.35, 0.075, 0.15)) ** 2,
};
process.stdout.write(JSON.stringify(out));
