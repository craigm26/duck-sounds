import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
page.on('pageerror', e => console.log('PAGEERROR', e.message));
await page.goto('http://127.0.0.1:8099/?v=' + Date.now(), { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
const info = await page.evaluate(() => {
  const w = window.__dbg;
  if (!w) return { error: 'no __dbg hook' };
  const { model, data, DUCK } = w;
  const rows = [];
  for (let g = 0; g < model.ngeom; g++) {
    const n = model.geom(g).name || ('g' + g);
    if (!/step|ball|block|cone|floor/.test(n)) continue;
    rows.push(`${n} type=${model.geom_type[g]} size=(${[0,1,2].map(i=>model.geom_size[g*3+i].toFixed(3))}) xpos=(${[0,1,2].map(i=>data.geom_xpos[g*3+i].toFixed(3))})`);
  }
  return { duckZ: data.qpos[DUCK.freeQpos+2], duckX: data.qpos[DUCK.freeQpos], ncon: data.ncon, rows: rows.slice(0, 10) };
});
console.log(JSON.stringify(info, null, 1).slice(0, 1800));
await browser.close();
