// Does the shipped page actually run in a browser? Loads it in headless
// Chromium, watches the console, waits for the simulator to report ready, then
// drives it forward and checks the duck moved.
import puppeteer from 'puppeteer-core';

const URL_ = (process.argv[2] || 'https://duck-craigmerry.pages.dev/') + '?v=' + Date.now();
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 900 });

const errors = [];
const logs = [];
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); else logs.push(m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('requestfailed', r => errors.push('requestfailed: ' + r.url() + ' ' + (r.failure()?.errorText || '')));

console.log('LOADING', URL_);
await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });

// the page adds .ready to <body> once physics + policy are loaded
let ready = false;
try {
  await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
  ready = true;
} catch { /* fall through and report status text */ }

const diag = await page.evaluate(() => ({
  title: document.title,
  hasStatus: !!document.getElementById('status'),
  status: document.getElementById('status')?.textContent.trim() ?? null,
  bodyClass: document.body.className,
})).catch(e => ({ error: String(e) }));
const status = diag.status ?? JSON.stringify(diag);
console.log('READY  :', ready);
console.log('STATUS :', JSON.stringify(status));

if (ready) {
  const hud0 = await page.$eval('#hud', el => el.textContent.trim());
  console.log('HUD t0 :', hud0);
  // hold ArrowUp for 6 s and see whether it walks
  await page.keyboard.down('ArrowUp');
  await new Promise(r => setTimeout(r, 6000));
  await page.keyboard.up('ArrowUp');
  const hud1 = await page.$eval('#hud', el => el.textContent.trim());
  console.log('HUD t6 :', hud1);
  const t0 = parseInt(hud0.match(/tick (\d+)/)?.[1] ?? '0', 10);
  const t1 = parseInt(hud1.match(/tick (\d+)/)?.[1] ?? '0', 10);
  console.log('TICKS  :', t0, '->', t1, `(${((t1 - t0) / 6).toFixed(1)} Hz, want ~50)`);
  // did anything actually get drawn?
  const box = await page.$eval('#view', el => { const r = el.getBoundingClientRect(); return { w: Math.round(r.width), h: Math.round(r.height) }; });
  console.log('CANVAS :', box.w + 'x' + box.h);
  await page.screenshot({ path: 'shot.png' });
  console.log('SHOT   : shot.png');
}
console.log('LOGS   :', logs.slice(0,6).join(' | '));
console.log('ERRORS :', errors.length ? '\n  ' + errors.slice(0, 12).join('\n  ') : 'none');
await browser.close();
