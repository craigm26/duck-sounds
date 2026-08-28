// Does the phone layout appear, and does the thumbstick actually drive?
import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?v=' + Date.now();
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });

for (const mode of ['phone', 'desktop']) {
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  if (mode === 'phone') {
    // hasTouch without isMobile: isMobile turns on Chromium's mobile viewport
    // override, which in this headless build collapses the page to 0x0. The
    // page keys off (pointer: coarse), which hasTouch alone provides.
    await page.setViewport({ width: 390, height: 844, hasTouch: true, deviceScaleFactor: 2 });
  } else {
    await page.setViewport({ width: 1200, height: 860 });
  }
  await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
  const layout = await page.evaluate(() => ({
    touchClass: document.body.classList.contains('is-touch'),
    stickVisible: document.getElementById('stick').offsetParent !== null,
    keysVisible: document.getElementById('keys').offsetParent !== null,
  }));
  let drove = null;
  if (mode === 'phone') {
    const box = await page.$eval('#stick', el => { const r = el.getBoundingClientRect();
      return { x: r.x + r.width/2, y: r.y + r.height/2, r: r.width/2 }; });
    await page.mouse.move(box.x, box.y);
    await page.mouse.down();
    await page.mouse.move(box.x, box.y - box.r * 0.8, { steps: 4 });
    await new Promise(r => setTimeout(r, 5000));
    const dbg = await page.evaluate(() => window.__stick ?? null);
    const hud = await page.$eval('#hud', el => el.textContent);
    console.log('   stickCmd =', JSON.stringify(dbg));
    await page.mouse.up();
    drove = parseFloat(hud.match(/speed ([\d.]+)/)[1]);
  }
  await page.screenshot({ path: `layout-${mode}.png` });
  console.log(`${mode.toUpperCase().padEnd(8)} touch=${layout.touchClass} stick=${layout.stickVisible} keys=${layout.keysVisible}` +
              (drove !== null ? `  stick drove at ${drove.toFixed(2)} m/s` : '') +
              `  errors=${errs.length || 'none'}`);
  await page.close();
}
await browser.close();
