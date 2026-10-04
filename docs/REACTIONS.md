# Reactions

The whole point of the app is that the reaction is a real cat or dog that clearly shows your mood. This file explains how reactions are chosen, the rules that limit how, and how to keep the phrase table accurate.

## Giphy's rules

From Giphy's developer best practices, and followed by both apps:

1. Search requests go directly from the browser to Giphy. No proxy for API calls or media. The page gets the key from `/api/config`.
2. No caching or storing of Giphy responses or media. The web keeps results for the current visit only; the desktop keeps them in memory for the session and deletes any Giphy files older versions left in `reaction_cache/`.
3. No reordering or filtering of Giphy's results. Results are used in the order Giphy returns them.
4. Attribution "Powered By GIPHY" wherever Giphy content shows.
5. Content rating `g`.

Rule 3 means bad results cannot be skipped in code. Precision comes only from which phrases are searched and how many of each phrase's top results are used.

## How the phrase table works

`MOOD_QUERIES` in `reactions.py` and `web/reactions.js` (identical, enforced by `tests/test_web_parity.py`) gives each mood one to three phrases per animal, each with a depth: how many of its top results to use. The app searches the first phrase with `limit` = depth and `offset` = 0, shows those results in Giphy's order, then searches the next phrase, and cycles. It never pages deeper into a phrase, because results are best at the top and drift to cartoons, people or other animals further down. A mood missing from the table falls back to "{mood} {animal}" with depth 8.

The table, as checked in October 2026:

| Mood | Cat phrases (top results used) | Dog phrases (top results used) |
|---|---|---|
| happy | cat smiling (12), happy cat (7) | dog smiling (12), happy dog (5) |
| sad | crying cat (11), sad cat (11) | sad dog (5), dog crying (2), depressed dog (1) |
| angry | angry cat (12) | angry dog (12) |
| surprise | shocked cat (12), surprised cat (12) | shocked dog (10), surprised dog (9) |
| fear | scared cat (12) | scared dog (9) |
| disgust | disgusted cat (7) | disgusted dog (7) |
| neutral | cat stare (11) | dog stare (12), dog staring (12) |
| smug | cat smirk (10) | smug dog (6) |
| confused | cat huh (12), confused cat (5) | dog head tilt (12), confused dog (12) |
| mischief | cat knocking things off (12), sneaky cat (9) | dog zoomies (10), guilty dog (2) |
| annoyed | annoyed cat (12), unimpressed cat (12), grumpy cat (12) | dog side eye (5), unimpressed dog (3), dog eye roll (2) |
| approval | cat thumbs up (3), cat smiling (12) | dog yes (4), dog thumbs up (2), dog smiling (12) |
| disapproval | cat side eye (9) | dog side eye (5), dog judging (6) |
| chill | relaxed cat (12) | lazy dog (10), sleepy dog (3) |
| suspicious | suspicious cat (5), cat side eye (9) | suspicious dog (2), dog side eye (5) |
| triumph | cat winning (3), cat smiling (12) | excited dog (12), dog winning (3) |
| anxious | scared cat (12) | nervous dog (5), scared dog (9) |
| bored | bored cat (12), cat yawning (11) | bored dog (12), dog yawning (6) |
| mocking | cat laughing (10), sassy cat (12) | dog grin (5), dog smiling (12) |
| stop | cat high five (3), cat no (4), cat says no (3) | dog high five (8) |
| determined | serious cat (12), cat butt wiggle (10) | serious dog (12) |
| focused | cat butt wiggle (10), serious cat (12) | serious dog (12), dog staring (12) |

## How it was chosen

Each candidate phrase was searched with the real key, and the top 12 results were laid out as contact sheets and judged by eye: real animal or not, right animal, mood readable or not. The winners were then checked again result by result at a larger size, because small thumbnails hide generated renders. About 540 results were checked this way (about 290 cat, 250 dog). Each depth stops just before the first bad result.

Phrases tried and rejected, so they are not retried blindly:

| Rejected | Problem |
|---|---|
| cat staring blankly, cat staring intently, cat staring at screen | Cartoons, people |
| excited cat, mad cat, cat hissing, disappointed cat | Cartoons mixed in early |
| cat dancing, cat victory, cat happy dance, cat celebrating | Many generated cats (jerseys, uniforms, hats, green screen) and cartoons |
| cat nodding approval, cat nodding, cat nod yes, cat approves, cat good job | Cartoons, people |
| disgusted cat past result 7, cat gross, cat gagging, cat grimace, cat disgusted face | People and TV characters |
| nervous cat, anxious cat, cat worried, cat trembling, cat hiding scared, worried kitten | Mostly cartoons |
| cat stop paw, cat paw up, cat paw stop, cat raising paw, talk to the paw cat | Stop motion, paw prints, people |
| determined cat, cat focused hunting, focused cat, cat concentrating, cat intense stare, cat hunting mode | Cartoons, people |
| proud cat, cat smug face, mischievous cat, cat suspicious look, cat squinting | Cartoons |
| puppy eyes, dog growling, dog smirk, dog disgusted face, dog hiding | Cartoons, people |
| dog nodding, dog nod, good boy | People, cartoons |
| grumpy dog | Returns cats |
| chill dog, relaxed dog, dog relaxing, dog sigh | Cartoons early |
| laughing dog, dog laughing, dog smug smile, sassy dog | Cartoons early |
| dog stealing food, dog caught in the act | People |
| focused dog, dog staring at ball, determined dog | People, cartoons |
| surprised animal, suspicious animal | Mostly cats anyway; adds nothing |

Specific bad results worth knowing: "sad dog" result 6 is Pixar's Dug (CGI), "dog yawning" result 7 is a cartoon, "cat thumbs up" result 4 looks generated.

## Re-checking

Giphy's results change over time, so audit every few months and after any complaint:

```
python dev/giphy_audit.py --key YOUR_KEY --animal cat
python dev/giphy_audit.py --key YOUR_KEY --animal dog
```

Each run makes one API call per distinct phrase (about 30 for cat, 35 for dog). The beta key's 100 calls an hour are shared with the live site, so do one animal per hour. Open the sheets in `dev/output/audit/`: green labels are results the app can show, red labels are the next few past the depth. If a green result is bad, lower that phrase's depth to stop before it. If the red ones are all good, the depth can go up. Change both `reactions.py` and `web/reactions.js`, then run `python tests/test_web_parity.py`.

To test a new phrase before adding it, add it to the table temporarily with a generous depth and audit just that mood with `--moods`.

## Backup photos

Used only while Giphy is not configured, rejects the key, or is failing (for example rate limited). Cats come from The Cat API (`/v1/images/search`, random, not mood matched). Dogs come from Dog CEO's per-breed endpoint over 26 pet breeds in `DOG_BREEDS` (same list in both apps), because its any-breed endpoint returns wild canids too.
