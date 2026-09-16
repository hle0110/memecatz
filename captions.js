// The static caption bank from captions.py. No API key, no network, no secrets.

export const STATIC_CAPTIONS = {
  happy: [["BIG MOOD", "NO NOTES"], ["THAT'S A YES", "FROM ME"]],
  sad: [["NOT VIBING", "RIGHT NOW"], ["PAUSE", "NOT OKAY"]],
  angry: [["SEND TWEET", ""], ["ABOUT TO", "SNAP"]],
  surprise: [["WAIT WHAT", ""], ["PLOT TWIST", "INCOMING"]],
  fear: [["NOPE NOPE NOPE", ""], ["RUN", "IT'S OVER"]],
  disgust: [["THE EW FACTOR", "IS HIGH"], ["HARD PASS", ""]],
  neutral: [["OKAY.", ""], ["PROCESSING", "..."]],
  smug: [["ROLL SAFE", "THINK ABOUT IT"], ["KNEW IT", "ALL ALONG"]],
  confused: [["WAIT", "WHAT JUST HAPPENED"], ["HOLD ON", "LET ME THINK"]],
  mischief: [["OH IT'S ON", ""], ["WATCH THIS", ""]],
  annoyed: [["HERE WE GO", "AGAIN"], ["NOT THIS", "AGAIN"]],
  approval: [["SEAL OF", "APPROVAL"], ["TAKE MY", "UPVOTE"]],
  disapproval: [["HARD NO", ""], ["ABSOLUTELY NOT", ""]],
  chill: [["ALL GOOD", "HERE"], ["VIBES ONLY", ""]],
  suspicious: [["SOMETHING'S", "NOT RIGHT"], ["I'M WATCHING", "YOU"]],
  triumph: [["NAILED IT", ""], ["W TAKEN", ""]],
  anxious: [["THIS IS FINE", "PROBABLY"], ["KEEP IT", "TOGETHER"]],
  bored: [["STILL WAITING", ""], ["ANY DAY NOW", ""]],
  mocking: [["SURE, BUDDY", ""], ["OKAY THERE", "CHAMP"]],
};

export function captionFor(primary) {
  const bank = STATIC_CAPTIONS[primary] || STATIC_CAPTIONS.neutral;
  const [top, bottom] = bank[Math.floor(Math.random() * bank.length)];
  return { top, bottom };
}
