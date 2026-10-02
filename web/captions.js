// The static caption bank, identical to STATIC_CAPTIONS in captions.py.
// tests/test_web_parity.py checks the two stay the same.

export const STATIC_CAPTIONS = {
  happy: [
    ["BIG MOOD", "NO NOTES"],
    ["THAT'S A YES", "FROM ME"],
    ["LIVING MY", "BEST LIFE"],
    ["TODAY IS", "A GOOD DAY"],
    ["SMILING", "FOR NO REASON"],
    ["PURE", "JOY"],
  ],
  sad: [
    ["NOT VIBING", "RIGHT NOW"],
    ["PAUSE", "NOT OKAY"],
    ["NEED A HUG", "IMMEDIATELY"],
    ["IT'S FINE", "IT'S NOT FINE"],
    ["CANCEL", "TODAY"],
    ["EMOTIONALLY", "BUFFERING"],
  ],
  angry: [
    ["SEND TWEET", ""],
    ["ABOUT TO", "SNAP"],
    ["WHO DID THIS", ""],
    ["I'M CALM", "I'M SO CALM"],
    ["DO NOT", "TEST ME"],
    ["COUNTING", "TO TEN"],
  ],
  surprise: [
    ["WAIT WHAT", ""],
    ["PLOT TWIST", "INCOMING"],
    ["DID THAT", "JUST HAPPEN"],
    ["HOLD", "THE PHONE"],
    ["EXCUSE ME", "??"],
    ["NOBODY SAW", "THAT COMING"],
  ],
  fear: [
    ["NOPE NOPE NOPE", ""],
    ["RUN", "IT'S OVER"],
    ["I HEARD", "A NOISE"],
    ["WHAT WAS", "THAT"],
    ["NOT TODAY", ""],
    ["HIDE", "EVERYTHING"],
  ],
  disgust: [
    ["THE EW FACTOR", "IS HIGH"],
    ["HARD PASS", ""],
    ["WHO MADE", "THIS"],
    ["SMELLS", "SUSPICIOUS"],
    ["I NEED", "A MINUTE"],
    ["NO THANK", "YOU"],
  ],
  neutral: [
    ["OKAY.", ""],
    ["PROCESSING", "..."],
    ["NO THOUGHTS", "HEAD EMPTY"],
    ["JUST", "EXISTING"],
    ["STARING", "INTO SPACE"],
    ["LOADING", "OPINION..."],
  ],
  smug: [
    ["ROLL SAFE", "THINK ABOUT IT"],
    ["KNEW IT", "ALL ALONG"],
    ["TOLD YOU", "SO"],
    ["TOO EASY", ""],
    ["CALCULATED", ""],
    ["I'M KIND OF", "A GENIUS"],
  ],
  confused: [
    ["WAIT", "WHAT JUST HAPPENED"],
    ["HOLD ON", "LET ME THINK"],
    ["I HAVE", "QUESTIONS"],
    ["THE MATH", "ISN'T MATHING"],
    ["SAY THAT", "AGAIN?"],
    ["HUH", ""],
  ],
  mischief: [
    ["OH IT'S ON", ""],
    ["WATCH THIS", ""],
    ["I HAVE", "A PLAN"],
    ["NOBODY", "TELL MOM"],
    ["HEHEHE", ""],
    ["CHAOS", "MODE ON"],
  ],
  annoyed: [
    ["HERE WE GO", "AGAIN"],
    ["NOT THIS", "AGAIN"],
    ["I'M SO TIRED", "OF THIS"],
    ["SIGH", ""],
    ["CAN WE", "NOT"],
    ["WHY IS IT", "ALWAYS ME"],
  ],
  approval: [
    ["SEAL OF", "APPROVAL"],
    ["TAKE MY", "UPVOTE"],
    ["WE LOVE", "TO SEE IT"],
    ["TEN OUT OF", "TEN"],
    ["CERTIFIED", "GOOD"],
    ["YES", "CHEF"],
  ],
  disapproval: [
    ["HARD NO", ""],
    ["ABSOLUTELY NOT", ""],
    ["I DON'T", "LIKE THIS"],
    ["DENIED", ""],
    ["THAT'S A NO", "FROM ME"],
    ["TRY", "AGAIN"],
  ],
  chill: [
    ["ALL GOOD", "HERE"],
    ["VIBES ONLY", ""],
    ["NO STRESS", ""],
    ["TAKING IT", "EASY"],
    ["COOL CALM", "COLLECTED"],
    ["JUST", "CHILLING"],
  ],
  suspicious: [
    ["SOMETHING'S", "NOT RIGHT"],
    ["I'M WATCHING", "YOU"],
    ["SUS", ""],
    ["I SEE", "WHAT YOU DID"],
    ["EXPLAIN", "YOURSELF"],
    ["TRUST", "NOBODY"],
  ],
  triumph: [
    ["NAILED IT", ""],
    ["W TAKEN", ""],
    ["CHAMPION", "ENERGY"],
    ["VICTORY", "IS MINE"],
    ["UNDEFEATED", ""],
    ["GG", ""],
  ],
  anxious: [
    ["THIS IS FINE", "PROBABLY"],
    ["KEEP IT", "TOGETHER"],
    ["OVERTHINKING", "AGAIN"],
    ["DID I", "LOCK THE DOOR"],
    ["DEEP", "BREATHS"],
    ["STRESS LEVEL", "HIGH"],
  ],
  bored: [
    ["STILL WAITING", ""],
    ["ANY DAY NOW", ""],
    ["IS IT OVER", "YET"],
    ["SO", "BORED"],
    ["NAP", "TIME?"],
    ["WAKE ME UP", "WHEN IT'S DONE"],
  ],
  mocking: [
    ["SURE, BUDDY", ""],
    ["OKAY THERE", "CHAMP"],
    ["WOW", "SO IMPRESSIVE"],
    ["THAT'S", "CUTE"],
    ["MHM", "SURE"],
    ["GOOD FOR", "YOU I GUESS"],
  ],
  stop: [
    ["STOP", "RIGHT THERE"],
    ["HOLD UP", ""],
    ["NOT ONE", "MORE STEP"],
    ["THAT'S", "ENOUGH"],
    ["PAUSE", ""],
    ["HALT", ""],
  ],
  determined: [
    ["LET'S", "GO"],
    ["LOCKED IN", ""],
    ["NOTHING", "CAN STOP ME"],
    ["GAME FACE", "ON"],
    ["WATCH ME", "WORK"],
    ["NO DAYS", "OFF"],
  ],
  focused: [
    ["LOCKED", "IN"],
    ["DO NOT", "DISTURB"],
    ["IN THE", "ZONE"],
    ["ONE JOB", ""],
    ["CONCENTRATING", ""],
    ["EYES ON", "THE PRIZE"],
  ],
};

// Picks a caption for the mood, avoiding an immediate repeat of the previous
// one when the bank has more than one option.
export function captionFor(primary, previous = null) {
  const bank = STATIC_CAPTIONS[primary] || STATIC_CAPTIONS.neutral;
  let options = bank;
  if (previous && bank.length > 1) {
    const filtered = bank.filter(([t, b]) => !(t === previous.top && b === previous.bottom));
    if (filtered.length) options = filtered;
  }
  const [top, bottom] = options[Math.floor(Math.random() * options.length)];
  return { top, bottom };
}
