// There is no controller plugged into this Pi, so the Gamepad API is stubbed
// and the page is driven through it. That exercises the real code path — poll,
// deadzone, EMA, edge detection, action dispatch — everything except the
// hardware itself.
import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?v=' + Date.now();
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 900 });
const errs = [];
page.on('pageerror', e => errs.push(e.message));
page.on('console', m => { if (m.type()==='error') errs.push(m.text()); });

await page.evaluateOnNewDocument(() => {
  window.__pad = { axes: [0, 0, 0, 0], buttons: new Array(17).fill(0), id: 'Stub Pad (STANDARD GAMEPAD)' };
  navigator.getGamepads = () => [{
    id: window.__pad.id, index: 0, connected: true, mapping: 'standard',
    axes: window.__pad.axes,
    buttons: window.__pad.buttons.map(v => ({ pressed: v > 0.5, touched: v > 0, value: v })),
  }];
});
await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
const wait = ms => new Promise(r => setTimeout(r, ms));
const hud = () => page.$eval('#hud', el => el.textContent);
const set = (axes, buttons) => page.evaluate((a, b) => {
  if (a) window.__pad.axes = a;
  window.__pad.buttons = new Array(17).fill(0);
  for (const i of (b || [])) window.__pad.buttons[i] = 1;
}, axes, buttons);

console.log('PAD state:', await page.$eval('#padState', el => el.textContent.trim()));

// 1. left stick forward
await page.keyboard.press('r'); await wait(500);
await set([0, -0.9, 0, 0], []);
await wait(6000);
const drove = parseFloat((await hud()).match(/speed ([\d.]+)/)[1]);
console.log(`PAD stick forward       speed ${drove.toFixed(2)} m/s  ${drove > 0.05 ? 'DRIVES' : 'no motion'}`);
await set([0, 0, 0, 0], []);

// 2. buttons fire intents. X (2) = sit toggle -> the duck should go down.
await page.keyboard.press('r'); await wait(700);
const before = parseFloat((await hud()).match(/height ([\d.]+)/)[1]);
await set([0,0,0,0], [2]); await wait(120); await set([0,0,0,0], []);
await wait(2600);
const after = parseFloat((await hud()).match(/height ([\d.]+)/)[1]);
console.log(`PAD X = sit             height ${before.toFixed(3)} -> ${after.toFixed(3)}  ${after < before - 0.02 ? 'SAT DOWN' : 'no change'}`);

// 3. a held button must fire ONCE, not every frame
await page.keyboard.press('r'); await wait(700);
const fires = await page.evaluate(async () => {
  let n = 0;
  const orig = console.debug;
  window.__count = () => n++;
  return new Promise(res => {
    const el = document.getElementById('padState');
    // count intent starts by watching the key row's active class
    let wasOn = false, count = 0;
    const iv = setInterval(() => {
      const on = !!document.querySelector('.keys button.on');
      if (on && !wasOn) count++;
      wasOn = on;
    }, 50);
    window.__pad.buttons = new Array(17).fill(0);
    window.__pad.buttons[4] = 1;            // hold LB (kick left)
    setTimeout(() => { window.__pad.buttons = new Array(17).fill(0); clearInterval(iv); res(count); }, 2500);
  });
});
console.log(`PAD held LB fired ${fires} time(s)  ${fires === 1 ? 'EDGE-TRIGGERED' : 'repeating — bug'}`);

// 4. remapping persists
const remapped = await page.evaluate(async () => {
  const rows = [...document.querySelectorAll('#padList .ctl')];
  const row = rows.find(r => r.querySelector('label').textContent.startsWith('Pick up'));
  row.querySelector('button').click();
  await new Promise(r => setTimeout(r, 120));
  window.__pad.buttons = new Array(17).fill(0);
  window.__pad.buttons[3] = 1;             // press Y while listening
  await new Promise(r => setTimeout(r, 400));
  window.__pad.buttons = new Array(17).fill(0);
  await new Promise(r => setTimeout(r, 200));
  return { shown: row.querySelector('.btnid').textContent,
           stored: JSON.parse(localStorage.getItem('microduck.padmap.v1') || '{}').ground_pick };
});
console.log(`PAD remap Pick up -> ${remapped.shown}, stored=${remapped.stored}  ${remapped.stored === 3 ? 'SAVED' : 'not saved'}`);
console.log('ERRORS:', errs.length ? errs.slice(0,3).join(' | ') : 'none');
await browser.close();
