// Switch to skates in a real browser and confirm it actually skates.
import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?v=' + Date.now();
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 860 });
const errs = [];
page.on('pageerror', e => errs.push(e.message));
page.on('console', m => { if (m.type()==='error') errs.push(m.text()); });
await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
const hud = () => page.$eval('#hud', el => el.textContent);
const wait = ms => new Promise(r => setTimeout(r, ms));

async function drive(label) {
  await page.keyboard.press('r'); await wait(500);
  await page.keyboard.down('ArrowUp');
  await wait(6000);
  const t = await hud();
  await page.keyboard.up('ArrowUp');
  const speed = parseFloat(t.match(/speed ([\d.]+)/)[1]);
  const h = parseFloat(t.match(/height ([\d.]+)/)[1]);
  console.log(`VARIANT ${label.padEnd(8)} speed ${speed.toFixed(2)} m/s   height ${h.toFixed(3)} m`);
  return speed;
}
const legs = await drive('legs');
await page.select('#variant', 'rollers');
await page.waitForFunction(() => document.getElementById('status').textContent === '', { timeout: 180000 });
await wait(600);
const rollers = await drive('skates');
await page.select('#variant', 'legs');
await page.waitForFunction(() => document.getElementById('status').textContent === '', { timeout: 180000 });
await wait(600);
const back = await drive('legs*');
console.log(`SWITCH skates/legs speed ratio ${(rollers / Math.max(legs, 0.01)).toFixed(2)}x`);
console.log('ERRORS:', errs.length ? errs.slice(0,3).join(' | ') : 'none');
await page.screenshot({ path: 'skates.png' });
await browser.close();
