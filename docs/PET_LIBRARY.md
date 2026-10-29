# Pet library

Every clip in the pet library is a real recording of a real cat or dog: your own pet, someone else's pet filmed with the owner's permission, or a clip published under a license that allows reuse without asking. Never generated, drawn, cartoon or 3D, never people, and never media saved from Giphy, Tenor, social media or any other site whose license does not allow it.

The library fills moods where Giphy has few good results (stop, approval, and for dogs sad and suspicious). A mood with at least one valid library item always shows library items and never searches Giphy. Every other mood uses Giphy exactly as before.

## Where it lives

`web/pets/library.json` lists the clips, and the clip files sit next to it under `web/pets/`. The site serves them from `/pets/`, and the desktop app reads the same file and folder from disk, so both apps always show the same clips. `tests/test_web_parity.py` checks that.

## Adding a clip

1. Put the original recording (your own, or a licensed download, see below) in `web/pets/source/`, which git ignores, so originals are never committed or published. Trim it to 2 to 4 seconds of the pet clearly showing the mood. No people in the frame.
2. Convert it to animated WebP, about 480 px wide, under 500 KB. Homebrew's ffmpeg has no WebP encoder, so this takes two steps: ffmpeg writes the frames, then `img2webp` (from `brew install webp`) joins them into a looping WebP:

   ```
   mkdir -p frames
   ffmpeg -i web/pets/source/stop_source.mov -ss 0 -t 3 -vf "fps=15,scale=480:-2" frames/%03d.png
   img2webp -loop 0 -lossy -q 60 -d 67 frames/*.png -o web/pets/dog/stop-1.webp
   rm -r frames
   ```

   `-ss` is where the clip starts and `-t` how long it runs. `-d 67` is one frame at 15 fps in milliseconds, so change it with `fps` (1000 divided by fps). If the file is 500 KB or more, lower `-q` or `fps`. `webpmux -info web/pets/dog/stop-1.webp` shows the frame count, each frame's duration and "Loop Count : 0" (loops forever).

   mp4 also works, but WebP is preferred, because an mp4 cannot go into the Daily challenge share image:

   ```
   ffmpeg -i web/pets/source/stop_source.mov -ss 0 -t 3 -vf scale=480:-2 -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart web/pets/dog/stop-1.mp4
   ```

3. Put the file in `web/pets/cat/` or `web/pets/dog/`.
4. Add an item under the animal and mood in `web/pets/library.json`:

   ```json
   { "file": "dog/stop-1.webp", "type": "webp", "credit": "Mochi, filmed by Hieu", "permission": "owner" }
   ```

   A licensed clip also names its page and license:

   ```json
   { "file": "dog/stop-2.webp", "type": "webp", "credit": "Jane Doe on Pexels", "permission": "licensed",
     "source": "https://www.pexels.com/video/...", "license": "Pexels License" }
   ```

5. Run `node --test tests/web_reactions.test.mjs` and `python tests/test_web_parity.py`. The first one fails if an item is malformed, its file is missing, or the file is 500 KB or more.

Items for a mood take turns in the order they are listed, and the one on screen is skipped when there is another.

## Fields

| Field | Value |
|---|---|
| `file` | Path inside `web/pets/`, such as `dog/stop-1.webp`. Letters, digits, `_`, `-`, `.` and `/` only, no `..`, and it must end in `.webp` or `.mp4` to match `type`. |
| `type` | `webp` or `mp4`. |
| `credit` | Who the pet is and who filmed it. Shown as "Pet clip: ..." on the reaction, in snapshots and in the share image. |
| `permission` | `owner` when it is your pet, `with permission` when the owner agreed to its use on this site, `licensed` when the clip was published under a license that allows reuse. |
| `source` | Licensed clips only: the `https` page the clip was downloaded from. |
| `license` | Licensed clips only: the license's name, such as `Pexels License`, `Pixabay Content License` or `CC0`. |

An item that breaks any of these rules is ignored by both apps.

## The permission rule

Only add a clip when one of these is true, and record which in `permission`:

1. `owner`: the pet is yours and you filmed it.
2. `with permission`: the owner said yes to this use on this site.
3. `licensed`: the clip's page shows a license that allows free reuse and editing without asking, such as the Pexels License, the Pixabay Content License or CC0. CC BY also works if `credit` names the creator as the license asks. Not allowed: "editorial use only", "no derivatives" or "non-commercial" licenses, clips with no license shown, and anything from Giphy, Tenor or social media.

For a licensed clip, open its page, read the license there, and check the clip yourself: a real cat or dog, no people, nothing generated or edited to look real. Name the creator and site in `credit` even when the license does not require it. If an owner or creator later asks for a clip to be removed, delete its file and its item in the same change.
