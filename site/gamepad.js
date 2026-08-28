// Bluetooth controller support, mapped the way the robot's own runtime maps it.
//
// The Microduck ships with a pad, so the sim should answer to the same buttons
// the robot does. The defaults below are Pollen's documented mapping from
// microduck_runtime (see their controls/gamepad.js): left stick drives, A is a
// ground pick, X and DpadDown toggle sit, RB and LB are the right and left
// kicks, DpadUp held for about a second swaps legs for skates.
//
// Everything is remappable, because "standard mapping" is a hopeful name — the
// same physical button lands on a different index depending on the pad, the
// browser and the OS, and a Bluetooth pad paired to a phone is the case most
// likely to disagree.
export const DEADZONE = 0.15;
const ALPHA = 0.12;          // EMA toward the stick target, as the runtime does
const HOLD_MS = 900;         // DpadUp hold that swaps the variant

/** Actions the pad can fire. Kept separate from the key map on purpose. */
export const ACTIONS = [
  { id: 'ground_pick', label: 'Pick up',      button: 0  },  // A
  { id: 'sit_toggle',  label: 'Sit / stand',  button: 2  },  // X
  { id: 'kick_left',   label: 'Kick left',    button: 4  },  // LB
  { id: 'kick_right',  label: 'Kick right',   button: 5  },  // RB
  { id: 'step_up',     label: 'Step up',      button: 7  },  // RT
  { id: 'roulade',     label: 'Forward roll', button: 6  },  // LT
  { id: 'hold',        label: 'Hold still',   button: 1  },  // B
  { id: 'reset',       label: 'Reset',        button: 9  },  // Start
  { id: 'variant',     label: 'Legs / skates', button: 12, hold: true }, // DpadUp, held
];

const STORE = 'microduck.padmap.v1';

export function loadMap() {
  const map = Object.fromEntries(ACTIONS.map(a => [a.id, a.button]));
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) || '{}');
    for (const k of Object.keys(map)) if (Number.isInteger(saved[k])) map[k] = saved[k];
  } catch { /* a corrupt or blocked store just means defaults */ }
  return map;
}

export function saveMap(map) {
  try { localStorage.setItem(STORE, JSON.stringify(map)); } catch { /* private mode */ }
}

export function resetMap() {
  try { localStorage.removeItem(STORE); } catch {}
  return Object.fromEntries(ACTIONS.map(a => [a.id, a.button]));
}

/**
 * Poll the pad. Call once per frame.
 *
 * `onAction(id)` fires on the RISING EDGE of a button, once per press — a pad
 * polled at 60 Hz reports a held button as pressed every frame, and without
 * edge detection a single press would fire an intent sixty times a second.
 */
export function makePad({ onAction, onConnect }) {
  let map = loadMap();
  const prev = new Map();
  const holdSince = new Map();
  let smoothX = 0, smoothY = 0;
  let listening = null;          // action id awaiting a button press
  let announced = false;

  const dz = v => (Math.abs(v) < DEADZONE ? 0 : v);

  function pads() {
    return (navigator.getGamepads ? [...navigator.getGamepads()] : []).filter(Boolean);
  }

  function poll(now) {
    const gp = pads()[0];
    if (!gp) { announced = false; return null; }
    if (!announced) { announced = true; onConnect?.(gp.id); }

    // Remapping: the next button pressed becomes this action's button.
    if (listening) {
      for (let b = 0; b < gp.buttons.length; b++) {
        if (gp.buttons[b]?.pressed) {
          map[listening.id] = b;
          saveMap(map);
          const done = listening; listening = null;
          done.resolve(b);
          break;
        }
      }
    }

    const lx = dz(gp.axes[0] ?? 0), ly = dz(gp.axes[1] ?? 0);
    smoothX += ALPHA * (lx - smoothX);
    smoothY += ALPHA * (ly - smoothY);

    for (const a of ACTIONS) {
      const idx = map[a.id];
      const down = !!gp.buttons[idx]?.pressed;
      const was = prev.get(a.id) || false;
      prev.set(a.id, down);
      if (listening) continue;                       // do not fire while remapping
      if (a.hold) {
        if (down && !was) holdSince.set(a.id, now);
        if (down && now - (holdSince.get(a.id) ?? now) >= HOLD_MS && !holdSince.get(a.id + ':fired')) {
          holdSince.set(a.id + ':fired', true);
          onAction(a.id);
        }
        if (!down) holdSince.delete(a.id + ':fired');
      } else if (down && !was) {
        onAction(a.id);
      }
    }
    // Forward is up, and up is negative on every pad.
    return { vx: -smoothY, vyaw: -smoothX, id: gp.id };
  }

  return {
    poll,
    get map() { return map; },
    listen(id) { return new Promise(resolve => { listening = { id, resolve }; }); },
    cancel() { listening = null; },
    reset() { map = resetMap(); return map; },
    connected: () => pads().length > 0,
  };
}
