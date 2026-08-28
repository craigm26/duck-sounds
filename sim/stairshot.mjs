import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 640 });
await page.goto('http://127.0.0.1:8099/?demo=1&v=' + Date.now(), { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
await page.evaluate(() => {
  const r = document.getElementById('rise'); r.value = 24; r.dispatchEvent(new Event('input', { bubbles: true }));
  const c = document.getElementById('count'); c.value = 5; c.dispatchEvent(new Event('input', { bubbles: true }));
});
// stand the duck at the foot of the stairs, which now live against the wall
await page.evaluate(() => window.__demo.place(0.30, 1.305 - 0.0));
await new Promise(r => setTimeout(r, 1500));
await (await page.$('#view')).screenshot({ path: 'stairs-wall.png' });
console.log('SHOT stairs-wall.png');
await browser.close();
