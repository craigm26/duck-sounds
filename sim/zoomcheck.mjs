// Zoom, on-screen button remapping, and the AR button's presence.
import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?v=' + Date.now();
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 900, hasTouch: true });
const errs = [];
page.on('pageerror', e => errs.push(e.message));
await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
const wait = ms => new Promise(r => setTimeout(r, ms));

// 1. zoom slider changes the rendered image
const shot = async () => (await (await page.$('#view')).screenshot({ encoding: 'base64' })).length;
await page.evaluate(() => { const e = document.getElementById('zoom'); e.value = 60; e.dispatchEvent(new Event('input')); });
await wait(700); const outSize = await shot();
await page.evaluate(() => { const e = document.getElementById('zoom'); e.value = 260; e.dispatchEvent(new Event('input')); });
await wait(700); const inSize = await shot();
console.log(`ZOOM  slider 0.6x vs 2.6x  frames differ: ${outSize !== inSize ? 'YES' : 'NO'}  (${outSize} vs ${inSize} bytes)`);
console.log(`ZOOM  readout now ${await page.$eval('#zoomOut', e => e.textContent)}`);

// 2. wheel zooms. Reset to the middle first: at the 3.2x ceiling a zoom-in
// is a no-op and the test would read as a failure.
await page.evaluate(() => { const e = document.getElementById('zoom'); e.value = 100; e.dispatchEvent(new Event('input')); });
await wait(200);
const before = await page.$eval('#zoomOut', e => e.textContent);
// page.mouse.wheel does not reach the canvas in this headless build — a
// harness limitation, not a page one — so the event is dispatched directly.
// That still exercises the real listener, preventDefault and all.
const after = await page.evaluate(() => {
  document.getElementById('view').dispatchEvent(
    new WheelEvent('wheel', { deltaY: -300, bubbles: true, cancelable: true }));
  return document.getElementById('zoomOut').textContent;
});
console.log(`WHEEL ${before} -> ${after}  ${before !== after ? 'ZOOMED' : 'no change'}`);

// 3. remap on-screen button A, and check it persists and fires the new intent
await page.select('#slotA', 'sit');
await wait(200);
const label = await page.$eval('#padA em', e => e.textContent);
const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('microduck.slots.v1') || '{}').A);
await page.keyboard.press('r'); await wait(600);
const h0 = await page.$eval('#hud', e => parseFloat(e.textContent.match(/height ([\d.]+)/)[1]));
await page.evaluate(() => document.getElementById('padA').dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
await wait(2800);
const h1 = await page.$eval('#hud', e => parseFloat(e.textContent.match(/height ([\d.]+)/)[1]));
console.log(`REMAP A -> "${label}" stored=${stored}  height ${h0.toFixed(3)} -> ${h1.toFixed(3)}  ${h1 < h0 - 0.02 ? 'FIRED SIT' : 'no change'}`);

// 4. AR button
// The button should appear only when an immersive session is actually
// available. Headless Chromium exposes navigator.xr but supports no device, so
// the correct behaviour here is HIDDEN — offering AR that cannot start is worse
// than not offering it.
const xr = await page.evaluate(async () => ({
  hidden: document.getElementById('xr').hidden,
  has: !!navigator.xr,
  ar: navigator.xr ? await navigator.xr.isSessionSupported('immersive-ar').catch(() => false) : false,
  vr: navigator.xr ? await navigator.xr.isSessionSupported('immersive-vr').catch(() => false) : false,
}));
const shouldShow = xr.ar || xr.vr;
console.log(`AR    xr=${xr.has} ar=${xr.ar} vr=${xr.vr}  button hidden=${xr.hidden}  ${shouldShow === !xr.hidden ? 'CORRECT' : 'MISMATCH'}`);
console.log('ERRORS:', errs.length ? errs.slice(0,3).join(' | ') : 'none');
await browser.close();
