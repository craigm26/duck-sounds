import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
// hasTouch without isMobile: isMobile turns on Chromium's mobile viewport
// override, which in this headless build collapses the page to 0x0. All the
// page actually keys off is (pointer: coarse), which hasTouch alone provides.
await page.setViewport({ width: 390, height: 844, hasTouch: true, deviceScaleFactor: 2 });
await page.goto('http://127.0.0.1:8099/?v=' + Date.now(), { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
console.log(JSON.stringify(await page.evaluate(() => {
  const q = id => {
    const el = document.getElementById(id);
    if (!el) return { missing: true };
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return { display: cs.display, visibility: cs.visibility, w: Math.round(r.width), h: Math.round(r.height),
             offsetParent: el.offsetParent ? el.offsetParent.tagName + '.' + el.offsetParent.className : null };
  };
  return {
    bodyClass: document.body.className,
    coarse: matchMedia('(pointer: coarse)').matches,
    touch: q('touch'), stick: q('stick'), view: q('view'),
    chain: (() => {
      const out = [];
      let el = document.getElementById('stick');
      while (el && el !== document.documentElement) {
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        out.push(`${el.tagName}.${el.className || '-'} display=${cs.display} w=${Math.round(r.width)} h=${Math.round(r.height)}`);
        el = el.parentElement;
      }
      return out;
    })(),
  };
}), null, 1));
await browser.close();
