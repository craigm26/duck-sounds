// Does each key actually produce the motion it claims? Presses it and watches
// the body, rather than trusting that a session loaded.
import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?v=' + Date.now();
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 800 });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type()==='error') errors.push(m.text()); });
await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });

const height = () => page.$eval('#hud', el => parseFloat(el.textContent.match(/height ([\d.]+)/)[1]));
const wait = ms => new Promise(r => setTimeout(r, ms));

// Height is the wrong probe for some of these: a kick is a leg motion, and
// "stand up" does nothing if the duck is already standing. So each case says
// what to watch and what has to happen first.
const speed = () => page.$eval('#hud', el => parseFloat(el.textContent.match(/speed ([\d.]+)/)[1]));
const CASES = [
  { key: 'c', label: 'Sit',          probe: 'height', want: 'drops' },
  { key: 'v', label: 'Stand up',     probe: 'height', want: 'rises', after: 'c' },
  { key: 'q', label: 'Kick left',    probe: 'speed',  want: 'moves' },
  { key: 'e', label: 'Kick right',   probe: 'speed',  want: 'moves' },
  { key: 'f', label: 'Pick up',      probe: 'height', want: 'dips' },
  { key: 'x', label: 'Forward roll', probe: 'height', want: 'dips' },
  { key: 'g', label: 'Step up',      probe: 'height', want: 'dips' },
  { key: 'b', label: 'Back roll',    probe: 'height', want: 'dips' },
];
for (const c of CASES) {
  await page.keyboard.press('r'); await wait(500);
  if (c.after) { await page.keyboard.press(c.after); await wait(3000); }
  const read = c.probe === 'speed' ? speed : height;
  const before = await read();
  await page.keyboard.press(c.key);
  await wait(1300);
  const during = await read();
  await wait(2400);
  const after = await read();
  const moved = Math.abs(during - before) > (c.probe === 'speed' ? 0.02 : 0.004)
             || Math.abs(after - before) > (c.probe === 'speed' ? 0.02 : 0.004);
  console.log(`INTENT ${c.key.toUpperCase()} ${c.label.padEnd(13)} ${c.probe} ${before.toFixed(3)} -> ${during.toFixed(3)} -> ${after.toFixed(3)}  ${moved ? 'MOVED' : 'NO CHANGE'}`);
}
console.log('ERRORS:', errors.length ? errors.slice(0,4).join(' | ') : 'none');
await page.screenshot({ path: 'intents.png' });
await browser.close();
