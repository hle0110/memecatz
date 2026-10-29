# Pet library

Every clip in the pet library is a real recording of your own pet, or of someone else's pet filmed with the owner's permission. Never generated, drawn, cartoon or 3D, never people, and never media saved from Giphy or any other site.

The library fills moods where Giphy has few good results (stop, approval, and for dogs sad and suspicious). A mood with at least one valid library item always shows library items and never searches Giphy. Every other mood uses Giphy exactly as before.

## Where it lives

`web/pets/library.json` lists the clips, and the clip files sit next to it under `web/pets/`. The site serves them from `/pets/`, and the desktop app reads the same file and folder from disk, so both apps always show the same clips. `tests/test_web_parity.py` checks that.

## Adding a clip

1. Trim the recording to 2 to 4 seconds of the pet clearly showing the mood. No people in the frame.
2. Convert it to animated WebP, about 480 px wide, under 500 KB. With ffmpeg:

   ```
   ffmpeg -i clip.mov -ss 0 -t 3 -vf "fps=15,scale=480:-2" -an -c:v libwebp -loop 0 -q:v 60 web/pets/dog/stop-1.webp
   ```

   Lower `-q:v` or `fps` if the file is over 500 KB. mp4 also works (`-an -c:v libx264 -pix_fmt yuv420p -movflags +faststart`, with a `.mp4` name), but WebP is preferred: an mp4 cannot go into the Daily challenge share image.
3. Put the file in `web/pets/cat/` or `web/pets/dog/`.
4. Add an item under the animal and mood in `web/pets/library.json`:

   ```json
   { "file": "dog/stop-1.webp", "type": "webp", "credit": "Mochi, filmed by Hieu", "permission": "owner" }
   ```

5. Run `node --test tests/web_reactions.test.mjs` and `python tests/test_web_parity.py`. The first one fails if an item is malformed, its file is missing, or the file is 500 KB or more.

Items for a mood take turns in the order they are listed, and the one on screen is skipped when there is another.

## Fields

| Field | Value |
|---|---|
| `file` | Path inside `web/pets/`, such as `dog/stop-1.webp`. Letters, digits, `_`, `-`, `.` and `/` only, no `..`, and it must end in `.webp` or `.mp4` to match `type`. |
| `type` | `webp` or `mp4`. |
| `credit` | Who the pet is and who filmed it. Shown as "Pet clip: ..." on the reaction, in snapshots and in the share image. |
| `permission` | `owner` when it is your pet, `with permission` when the owner agreed to its use on this site. |

An item that breaks any of these rules is ignored by both apps.

## The permission rule

Only add a clip when you own the pet or the owner has said yes to this use, and record which one in `permission`. If an owner later asks for a clip to be removed, delete its file and its item in the same change.
