// Does the shipped page actually run in a browser? Loads it in headless
// Chromium, watches the console, waits for the simulator to report ready, then
// drives it forward and checks the duck moved.
import puppeteer from 'puppeteer-core';

const URL_ = process.argv[2] || 'https://duck-craigmerry.pages.dev/';
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 900 });

const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
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
  const ink = await page.evaluate(() => {
    const c = document.getElementById('view');
    const g = c.getContext('2d');
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 8) n++;
    return { painted: n, total: d.length / 4 };
  });
  console.log('CANVAS :', ink.painted, 'of', ink.total, 'pixels painted',
              `(${(100 * ink.painted / ink.total).toFixed(1)}%)`);
  await page.screenshot({ path: 'shot.png' });
  console.log('SHOT   : shot.png');
}
console.log('ERRORS :', errors.length ? '\n  ' + errors.slice(0, 12).join('\n  ') : 'none');
await browser.close();
