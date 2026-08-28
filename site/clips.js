// The intent gallery.
//
// These were thirteen animated GIFs: 20 MB of them, all decoding at once,
// every one of them running whether or not it was on screen. As H.264 with a
// poster frame the same thirteen clips are 1.2 MB, and a video can be told to
// stop when it scrolls away — which a GIF cannot.
//
// Two rules:
//   play only what is on screen, so the tab is decoding one or two clips
//   rather than thirteen;
//   and if the visitor has asked for less motion, play nothing until they ask
//   for it — the poster frame is a real frame of the move, so the gallery
//   still reads.
const clips = [...document.querySelectorAll('.clips video')];
if (clips.length) {
  const still = matchMedia('(prefers-reduced-motion: reduce)');

  // Reduced motion turns each clip into a click-to-play still. The control has
  // to be a real button, not a click handler on the video, or it is unreachable
  // from the keyboard.
  function makeManual(v) {
    const btn = document.createElement('button');
    btn.className = 'mini clip-play';
    btn.type = 'button';
    const label = v.getAttribute('aria-label') || 'this clip';
    const paint = () => {
      btn.textContent = v.paused ? 'play' : 'pause';
      btn.setAttribute('aria-label', (v.paused ? 'play ' : 'pause ') + label);
    };
    btn.addEventListener('click', () => { v.paused ? v.play().catch(() => {}) : v.pause(); paint(); });
    v.addEventListener('play', paint);
    v.addEventListener('pause', paint);
    paint();
    v.after(btn);
    return btn;
  }

  let io = null;
  function auto(on) {
    if (io) { io.disconnect(); io = null; }
    if (!on) { for (const v of clips) v.pause(); return; }
    io = new IntersectionObserver(entries => {
      for (const e of entries) {
        const v = e.target;
        if (e.isIntersecting) {
          // preload="none" means the first play() is also the first fetch.
          v.play().catch(() => { /* autoplay refused: the poster stands in */ });
        } else {
          v.pause();
        }
      }
    }, { rootMargin: '150px 0px' });
    for (const v of clips) io.observe(v);
  }

  const buttons = [];
  function apply() {
    const manual = still.matches;
    if (manual && !buttons.length) for (const v of clips) buttons.push(makeManual(v));
    for (const b of buttons) b.hidden = !manual;
    auto(!manual);
  }
  apply();
  still.addEventListener('change', apply);
}
