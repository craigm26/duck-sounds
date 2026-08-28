// Shoot the page at every viewport that matters and report the geometry that
// the layout is supposed to guarantee, so "the desktop UI is right" is a
// measurement rather than an impression.
//
//   node layoutshots.mjs [url] [outdir]
//
// The numbers printed are the ones the redesign is judged on: how much of the
// window the simulator actually occupies, whether the controls sit over the
// canvas or below it, and whether anything overflows sideways.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?v=' + Date.now();
const OUT = process.argv[3] || 'shots';
fs.mkdirSync(OUT, { recursive: true });

const VIEWPORTS = [
  { name: '2560x1440', width: 2560, height: 1440 },
  { name: '1920x1080', width: 1920, height: 1080 },
  { name: '1440x900', width: 1440, height: 900 },
  { name: '1280x800', width: 1280, height: 800 },
  { name: '1024x768', width: 1024, height: 768 },
  { name: '900x700', width: 900, height: 700 },
  { name: '768x1024', width: 768, height: 1024 },
  { name: 'phone-390x844', width: 390, height: 844, hasTouch: true, deviceScaleFactor: 2 },
  { name: 'phone-land-844x390', width: 844, height: 390, hasTouch: true, deviceScaleFactor: 2 },
];

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'],
});

const rows = [];
for (const vp of VIEWPORTS) {
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  page.on('requestfailed', r => errs.push('REQ ' + r.url().split('/').pop()));
  await page.setViewport(vp);
  await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });
  try {
    await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
  } catch { errs.push('NEVER READY'); }
  await new Promise(r => setTimeout(r, 1200));

  const m = await page.evaluate(() => {
    const box = s => {
      const el = document.querySelector(s);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height),
               shown: el.offsetParent !== null || getComputedStyle(el).position === 'fixed' };
    };
    const view = document.getElementById('view');
    const cv = view ? view.getBoundingClientRect() : null;
    return {
      vw: innerWidth, vh: innerHeight,
      view: box('#view'),
      buffer: view ? { w: view.width, h: view.height } : null,
      // The headline number: how much of the window is simulator.
      coverage: cv ? +((cv.width * cv.height) / (innerWidth * innerHeight) * 100).toFixed(1) : null,
      stage: box('.stage'),
      hud: box('#hud'),
      keys: box('#keys'),
      pad: box('.pad'),
      controls: box('.controls'),
      touchClass: document.body.classList.contains('is-touch'),
      stickShown: (() => { const e = document.getElementById('stick'); return !!e && e.offsetParent !== null; })(),
      // Sideways overflow is always a bug.
      hScroll: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      docH: document.documentElement.scrollHeight,
    };
  });

  // Is anything that calls itself a control actually ON the canvas?
  const overlaid = m.view && ['keys', 'pad', 'controls'].filter(k => {
    const c = m[k];
    return c && c.shown && c.y < m.view.y + m.view.h && c.y + c.h > m.view.y;
  });

  rows.push({ vp: vp.name, ...m, overlaid });
  await page.screenshot({ path: `${OUT}/${vp.name}.png` });
  console.log(
    `${vp.name.padEnd(18)} view ${String(m.view?.w).padStart(5)}x${String(m.view?.h).padEnd(5)}` +
    ` buf ${String(m.buffer?.w).padStart(5)}x${String(m.buffer?.h).padEnd(5)}` +
    ` cover ${String(m.coverage).padStart(5)}%  doc ${String(m.docH).padStart(5)}` +
    `  hScroll ${m.hScroll}  touch ${m.touchClass ? 'Y' : 'n'}` +
    `  overlaid[${overlaid ? overlaid.join(',') : ''}]` +
    (errs.length ? `  ERRORS: ${errs.slice(0, 3).join(' | ')}` : ''));
  await page.close();
}

fs.writeFileSync(`${OUT}/layout.json`, JSON.stringify(rows, null, 2));
console.log(`\nwrote ${OUT}/layout.json and ${VIEWPORTS.length} screenshots`);
await browser.close();
