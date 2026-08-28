// The parts of the redesign a screenshot cannot check.
//
// Every assertion here is something that was either broken before or is newly
// possible to break: the keyboard used to hijack the sliders, the docks used
// not to exist, and the camera shear is a number that has to come out right or
// the duck sits behind the rail.
//
//   node uicheck.mjs [url]
import puppeteer from 'puppeteer-core';

const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/');
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'],
});

const results = [];
const check = (name, pass, detail = '') => {
  results.push({ name, pass, detail });
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

async function open(opts = {}) {
  const page = await browser.newPage();
  page.on('pageerror', e => check('no page errors', false, e.message));
  await page.setViewport(opts.viewport || { width: 1600, height: 1000 });
  if (opts.media) await page.emulateMediaFeatures(opts.media);
  await page.goto(URL_ + '?v=' + Date.now(), { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
  await new Promise(r => setTimeout(r, 800));
  return page;
}

// ── 1. the desktop instrument layout ───────────────────────────────────────
{
  const page = await open();

  const geom = await page.evaluate(() => {
    const r = s => { const e = document.querySelector(s); const b = e.getBoundingClientRect();
                     return { l: b.left, t: b.top, r: b.right, b: b.bottom, w: b.width, h: b.height }; };
    return { view: r('#view'), rail: r('.rail'), hud: r('#hud'),
             drive: r('.dock-drive'), tools: r('.dock-tools') };
  });
  const over = (a, b) => a.l < b.r && a.r > b.l && a.t < b.b && a.b > b.t;
  check('rail sits ON the canvas', over(geom.rail, geom.view));
  check('hud sits ON the canvas', over(geom.hud, geom.view));
  check('drive dock sits ON the canvas', over(geom.drive, geom.view));
  check('tools sit ON the canvas', over(geom.tools, geom.view));
  check('drive dock and rail do not collide', !over(geom.drive, geom.rail));
  check('hud and rail do not collide', !over(geom.hud, geom.rail));
  check('tools sit above the rail', geom.tools.b <= geom.rail.t + 1,
        `tools bottom ${geom.tools.b.toFixed(0)} vs rail top ${geom.rail.t.toFixed(0)}`);

  // ── 2. the camera shear puts the duck in the VISIBLE middle ─────────────
  // Measured differentially, on purpose. The absolute centroid of the duck's
  // gold is not the centre of the duck — the head sits forward and right of
  // the body — but the shear moves EVERY pixel by the same amount, so the
  // distance the duck travels when the rail opens is a clean read on it.
  //
  // readPixels is no use here: the context is not preserveDrawingBuffer, so
  // after compositing it returns nothing. drawImage onto a 2D canvas inside a
  // rAF does work, and that is what this uses.
  await page.keyboard.press('z');                    // hold still: no walking between the two reads
  await new Promise(r => setTimeout(r, 1200));
  const gold = () => page.evaluate(() => new Promise(res => {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const cv = document.getElementById('view');
      const c = document.createElement('canvas');
      c.width = cv.width; c.height = cv.height;
      const g = c.getContext('2d');
      g.drawImage(cv, 0, 0);
      const H = Math.round(c.height * 0.8);          // the coloured props live in the bottom band
      const d = g.getImageData(0, 0, c.width, H).data;
      let sum = 0, n = 0;
      for (let y = 0; y < H; y += 2) for (let x = 0; x < c.width; x += 2) {
        const i = (y * c.width + x) * 4;
        if (d[i + 3] < 8) continue;
        const r = d[i], gg = d[i + 1], b = d[i + 2];
        if (r > 110 && gg > 80 && b < gg * 0.72 && r - b > 55) { sum += x; n++; }
      }
      const rail = document.querySelector('.rail').getBoundingClientRect();
      const view = cv.getBoundingClientRect();
      const shown = getComputedStyle(document.querySelector('.rail')).display !== 'none';
      res({ n, x: n ? sum / n / c.width : null,
            occl: shown && rail.width ? Math.max(0, view.right - rail.left) / view.width : 0 });
    }));
  }));
  const openRail = await gold();
  await page.evaluate(() => document.getElementById('railToggle').click());
  await new Promise(r => setTimeout(r, 500));
  const shutRail = await gold();
  await page.evaluate(() => document.getElementById('railToggle').click());
  await new Promise(r => setTimeout(r, 300));
  if (openRail.x !== null && shutRail.x !== null) {
    const moved = shutRail.x - openRail.x, want = openRail.occl / 2;
    check('the camera shears by exactly what the rail covers',
          Math.abs(moved - want) < 0.02 && want > 0.05,
          `duck moves ${(moved * 100).toFixed(1)}% of the canvas when the rail opens, rail covers ${(openRail.occl * 100).toFixed(1)}% so ${(want * 100).toFixed(1)}% was wanted`);
  } else {
    check('the camera shears by exactly what the rail covers', false, 'could not read the canvas back');
  }

  // ── 3. the keyboard no longer hijacks the controls ──────────────────────
  const slider = await page.evaluate(() => {
    const el = document.getElementById('rise');
    el.focus();
    return el.value;
  });
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  const after = await page.$eval('#rise', el => el.value);
  check('arrow keys adjust a focused slider', after !== slider, `${slider} -> ${after}`);
  const drove = await page.evaluate(() => window.__cmdProbe);
  check('...and do not also walk the duck', drove === undefined || drove === null);

  // A select's type-ahead must not fire intents letter by letter.
  const fired = await page.evaluate(async () => {
    const sel = document.getElementById('variant');
    sel.focus();
    const before = document.querySelectorAll('.keys button.on').length;
    for (const k of ['s', 'c', 'v', 'b']) {
      sel.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
    }
    await new Promise(r => setTimeout(r, 250));
    return { before, after: document.querySelectorAll('.keys button.on').length };
  });
  check('typing in a <select> does not fire intents', fired.after === fired.before,
        `${fired.before} -> ${fired.after} active`);

  // ── 4. rail toggle and its persistence ──────────────────────────────────
  const toggled = await page.evaluate(async () => {
    const t = document.getElementById('railToggle');
    t.click();
    await new Promise(r => setTimeout(r, 120));
    const hidden = getComputedStyle(document.querySelector('.rail')).display === 'none';
    const stored = localStorage.getItem('microduck.rail.v1');
    t.click();
    await new Promise(r => setTimeout(r, 120));
    return { hidden, stored, backAgain: getComputedStyle(document.querySelector('.rail')).display !== 'none' };
  });
  check('rail toggle hides the rail', toggled.hidden);
  check('rail toggle is remembered', toggled.stored === '0');
  check('rail toggle restores the rail', toggled.backAgain);

  // ── 5. frame rate with a viewport-sized canvas ──────────────────────────
  const fps = await page.evaluate(() => new Promise(res => {
    let n = 0; const t0 = performance.now();
    const tick = () => { n++; performance.now() - t0 < 3000 ? requestAnimationFrame(tick)
                                                           : res(n / ((performance.now() - t0) / 1000)); };
    requestAnimationFrame(tick);
  }));
  check('frame rate is usable at a full-viewport canvas', fps > 20, `${fps.toFixed(1)} fps (software GL)`);

  // ── 6. the gallery is video, lazy, and only plays what is visible ───────
  const gal = await page.evaluate(async () => {
    const v = [...document.querySelectorAll('.clips video')];
    const imgs = document.querySelectorAll('.clips img').length;
    const playingBefore = v.filter(x => !x.paused).length;
    document.querySelector('.clips').scrollIntoView();
    await new Promise(r => setTimeout(r, 1500));
    return { count: v.length, imgs, playingBefore, playingAfter: v.filter(x => !x.paused).length,
             posters: v.filter(x => x.poster).length };
  });
  check('gallery is thirteen <video> clips, no GIFs', gal.count === 13 && gal.imgs === 0,
        `${gal.count} videos, ${gal.imgs} imgs`);
  check('every clip has a poster frame', gal.posters === gal.count);
  check('nothing plays before the gallery is on screen', gal.playingBefore === 0);
  check('clips play once scrolled to', gal.playingAfter > 0, `${gal.playingAfter} playing`);

  await page.close();
}

// ── 7. dark mode ───────────────────────────────────────────────────────────
{
  const page = await open({ media: [{ name: 'prefers-color-scheme', value: 'dark' }] });
  const dark = await page.evaluate(() => {
    const cs = getComputedStyle(document.body);
    const rail = document.querySelector('.rail .card');
    const rcs = getComputedStyle(rail);
    return { bg: cs.backgroundColor, fg: cs.color,
             cardBorder: rcs.borderTopColor, cardBg: rcs.backgroundColor,
             hudBorder: getComputedStyle(document.getElementById('hud')).borderTopColor };
  });
  const isDark = c => {
    const [r, g, b] = c.match(/\d+/g).map(Number);
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 90;
  };
  check('dark mode paints a dark page', isDark(dark.bg), dark.bg);
  // The bug this catches: color-mix with a >100% percentage is invalid, the
  // custom property becomes garbage at use time, and every dock border silently
  // falls back to currentColor — which in dark mode is near-white.
  check('dock borders survive dark mode', !/^rgba?\(2[0-4]\d|^rgba?\(25[0-5]/.test(dark.cardBorder),
        `card border ${dark.cardBorder}`);
  await page.close();
}

// ── 8. reduced motion ──────────────────────────────────────────────────────
{
  const page = await open({ media: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
  const rm = await page.evaluate(async () => {
    document.querySelector('.clips').scrollIntoView();
    await new Promise(r => setTimeout(r, 1200));
    const v = [...document.querySelectorAll('.clips video')];
    return { playing: v.filter(x => !x.paused).length, buttons: document.querySelectorAll('.clip-play').length };
  });
  check('reduced motion leaves the clips still', rm.playing === 0, `${rm.playing} playing`);
  check('reduced motion offers a play control per clip', rm.buttons === 13, `${rm.buttons} buttons`);
  await page.close();
}

// ── 9. the phone path still works ──────────────────────────────────────────
{
  const page = await open({ viewport: { width: 390, height: 844, hasTouch: true, deviceScaleFactor: 2 } });
  const phone = await page.evaluate(() => ({
    touch: document.body.classList.contains('is-touch'),
    stick: document.getElementById('stick').offsetParent !== null,
    pads: document.querySelectorAll('.pad-btn').length,
    movesHidden: getComputedStyle(document.getElementById('movesCard')).display === 'none',
    slotShown: getComputedStyle(document.getElementById('slotCard')).display !== 'none',
    arrowsHidden: getComputedStyle(document.querySelector('.dock-drive .pad')).display === 'none',
    hudWraps: getComputedStyle(document.getElementById('hud')).flexWrap === 'wrap',
  }));
  check('phone gets the touch layout', phone.touch && phone.stick && phone.pads === 2);
  check('phone hides the keyboard move list', phone.movesHidden);
  check('phone shows the A/B assignment card', phone.slotShown);
  check('phone hides the arrow pad', phone.arrowsHidden);
  check('phone HUD wraps rather than scrolling', phone.hudWraps);
  await page.close();
}

await browser.close();
const failed = results.filter(r => !r.pass);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
if (failed.length) { console.log('FAILED:'); for (const f of failed) console.log('  ' + f.name + '  ' + f.detail); }
process.exit(failed.length ? 1 : 0);
