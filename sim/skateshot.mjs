import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 700 });
await page.goto('http://127.0.0.1:8099/?v=' + Date.now(), { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
await page.select('#variant', 'rollers');
await page.waitForFunction(() => document.getElementById('status').textContent === '', { timeout: 180000 });
await new Promise(r => setTimeout(r, 2500));
const el = await page.$('#view');
await el.screenshot({ path: 'skates.png' });
console.log('SHOT skates.png');
await browser.close();
