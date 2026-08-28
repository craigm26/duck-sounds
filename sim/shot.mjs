import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?v=' + Date.now();
const rise = process.argv[3] || '12';
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 820 });
await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
// set a visible staircase
await page.evaluate(r => {
  const el = document.getElementById('rise');
  el.value = r; el.dispatchEvent(new Event('input', { bubbles: true }));
}, rise);
await new Promise(r => setTimeout(r, 1500));
await page.screenshot({ path: 'room.png' });
console.log('SHOT room.png with rise =', rise, 'mm');
await browser.close();
