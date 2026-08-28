// Diff the browser's state against the search's, tick by tick, on the wall
// flip. If ctrl matches and the body still diverges it is physics; if ctrl
// differs, the move is being played differently.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const NEARGAP = JSON.parse(fs.readFileSync('wallflip-best.json','utf8')).p.startGap;

// --- browser side
const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 700, height: 480 });
await page.goto('http://127.0.0.1:8099/?demo=1&v=' + Date.now(), { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
const wait = ms => new Promise(r => setTimeout(r, ms));
await page.evaluate(g => window.__demo.place(1.5 - 0.05 - g), NEARGAP);
await page.evaluate(a => window.__demo.settle(25, a), 0.07513);
await wait(150);
await page.evaluate(() => window.__demo.record(12));
await page.keyboard.press('t');
await wait(1200);
const web = await page.evaluate(() => window.__demo.dump());
await browser.close();

// --- node side
const { attemptTrace } = await import('./wallflip_trace.mjs');
const node = await attemptTrace(JSON.parse(fs.readFileSync('wallflip-best.json','utf8')).p, 12);

console.log('browser ticks captured:', web.length, ' node ticks:', node.length);
const n = Math.min(web.length, node.length);
for (let i = 0; i < Math.min(n, 6); i++) {
  const dObs = Math.max(...web[i].obs.map((v, k) => Math.abs(v - node[i].obs[k])));
  const dCtrl = Math.max(...web[i].ctrl.map((v, k) => Math.abs(v - node[i].ctrl[k])));
  console.log(`  t${i}  max|obs diff| ${dObs.toFixed(5)}   max|ctrl diff| ${dCtrl.toFixed(5)}   z web ${web[i].z} node ${node[i].z}`);
}
// where in the observation do they differ most, at the first tick?
if (n) {
  const d = web[0].obs.map((v, k) => Math.abs(v - node[0].obs[k]));
  const worst = d.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]).slice(0, 6);
  const block = k => k < 3 ? 'gyro' : k < 6 ? 'gravity' : k < 20 ? 'joint_pos' : k < 34 ? 'joint_vel' : k < 48 ? 'last_action' : 'command';
  console.log('  biggest first-tick differences:', worst.map(([v, k]) => `${block(k)}[${k}]=${v.toFixed(4)}`).join('  '));
}
