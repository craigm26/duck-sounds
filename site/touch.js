// Touch controls: a thumbstick and two action pads, for phones.
//
// The layout switches on the POINTER, not the viewport width. A narrow desktop
// window is still a keyboard; an iPad in landscape is still a thumb. Matching
// `(pointer: coarse)` gets that right where a width breakpoint does not.
export function isTouch() {
  return window.matchMedia('(pointer: coarse)').matches;
}

/**
 * A thumbstick. Reports a unit vector; the caller decides what it means.
 * Anchors wherever the thumb lands rather than at a fixed centre, which is what
 * stops a stick feeling like it is fighting you when you grab it off-centre.
 */
export function makeStick(el, onMove) {
  const knob = el.querySelector('.knob');
  let id = null, cx = 0, cy = 0;
  const radius = () => el.clientWidth / 2 - 14;

  const set = (dx, dy) => {
    const r = radius();
    const len = Math.hypot(dx, dy);
    const s = len > r ? r / len : 1;
    knob.style.transform = `translate(${dx * s}px, ${dy * s}px)`;
    onMove(dx * s / r, dy * s / r);
  };
  const release = () => {
    id = null;
    knob.style.transform = 'translate(0px, 0px)';
    onMove(0, 0);
  };

  el.addEventListener('pointerdown', e => {
    if (id !== null) return;
    id = e.pointerId;
    el.setPointerCapture(id);
    const r = el.getBoundingClientRect();
    cx = r.left + r.width / 2; cy = r.top + r.height / 2;
    set(e.clientX - cx, e.clientY - cy);
    e.preventDefault();
  });
  el.addEventListener('pointermove', e => {
    if (e.pointerId !== id) return;
    set(e.clientX - cx, e.clientY - cy);
    e.preventDefault();
  });
  for (const ev of ['pointerup', 'pointercancel', 'pointerleave']) {
    el.addEventListener(ev, e => { if (e.pointerId === id) release(); });
  }
  return { release };
}
