// Helper for test_web_parity.py. Reads a JSON request on stdin, runs the
// browser modules from web/ in Node, and prints their results as JSON.
import { readFileSync } from "node:fs";
import { combine, topTags, AU_TAG_TO_MOOD } from "../web/mood.js";
import { STATIC_CAPTIONS } from "../web/captions.js";
import {
  MOOD_QUERIES, DOG_BREEDS, queriesForMood, PET_LIBRARY_URL, PET_LIBRARY_DIR, PET_TYPES, PET_PERMISSIONS,
  libraryEntries, nextLibraryItem, petCreditLine,
} from "../web/reactions.js";
import { classifyGesturePoints, GESTURE_TO_TAGS, FEATURE_KEYS, rateAdjustedAlpha } from "../web/face.js";
import { toSquare, cropRect, grayOf } from "../web/emotion.js";

const req = JSON.parse(readFileSync(0, "utf8"));

// Walks a library the way pick() does: each step asks for one mood, skipping
// the key on screen. An empty list for a mood means it falls back to Giphy.
function libraryWalk({ library, animal, steps }) {
  const cursor = {};
  let onScreen = null;
  return steps.map((mood) => {
    const [item, next] = nextLibraryItem(libraryEntries(library, mood, animal), cursor[mood] || 0, onScreen);
    if (!item) return "giphy";
    cursor[mood] = next;
    onScreen = item.key;
    return item;
  });
}

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
  petLibraryUrl: PET_LIBRARY_URL,
  petLibraryDir: PET_LIBRARY_DIR,
  petTypes: PET_TYPES,
  petPermissions: PET_PERMISSIONS,
  petCredit: petCreditLine("Mochi, filmed by the owner"),
  petLibraryFile: JSON.parse(readFileSync(new URL(`../web${PET_LIBRARY_URL}`, import.meta.url), "utf8")),
  petWalks: req.libraryCases.map(libraryWalk),
  petFileWalks: req.fileWalkCases.map((c) => libraryWalk({ ...c, library: JSON.parse(readFileSync(new URL(`../web${PET_LIBRARY_URL}`, import.meta.url), "utf8")) })),
  alphaSame: rateAdjustedAlpha(0.35, 0.15, 0.15),
  alphaTwoHalves: 1 - (1 - rateAdjustedAlpha(0.35, 0.075, 0.15)) ** 2,
};
process.stdout.write(JSON.stringify(out));
