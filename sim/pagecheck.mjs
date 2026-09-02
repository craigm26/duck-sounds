// Open site/phonebench in a real browser and print what the page printed.
//
// WHY IT IS HERE. `phonebench.html` is the spike — the page whose whole job is
// to answer, on the operator's own iPhone, whether Safari runs this. Nothing on
// this Pi can be that iPhone. What this CAN do is prove the page is not broken
// before it is deployed: that its module graph resolves, that no request 404s,
// that nothing throws, and that the physics-parity line actually appears with a
// number in it. A spike that fails on the phone for a reason a Pi could have
// caught costs a round trip through a deploy and somebody else's afternoon.
//
// It found two things on its first run: the JS-engine sniff in
// `duckbench-web.mjs` reported JavaScriptCore for Chromium, and the "one tick,
// end to end" row was timed cold and included fetching a 791 KB policy.
//
// Chromium here is NOT Safari and the numbers it prints are not the phone's.
// Read them as an upper bound on brokenness, never as a measurement of iOS.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'site', 'phonebench');
if (!fs.existsSync(ROOT)) { console.error('run scripts/make_phonebench.sh first'); process.exit(2); }
const CHROME = process.env.CHROME || '/usr/bin/chromium';
const TYPES = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json',
                '.wasm': 'application/wasm', '.html': 'text/html' };
const missing = [];
const server = http.createServer((req, res) => {
  let p = new URL(req.url, 'http://x').pathname;
  if (p === '/') p = '/index.html';
  const file = path.join(ROOT, p);
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    missing.push(p); res.writeHead(404); return res.end('no');
  }
  // The same three headers site/_headers sets, so this serves what Pages will.
  res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream',
                       'cross-origin-opener-policy': 'same-origin',
                       'cross-origin-embedder-policy': 'require-corp',
                       'cross-origin-resource-policy': 'same-origin' });
  fs.createReadStream(file).pipe(res);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}/`;

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage'] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e.message)));
await page.goto(base, { waitUntil: 'load', timeout: 120000 });
// The page does its work in a top-level await after load; wait for the verdict
// row rather than for a fixed number of seconds.
await page.waitForFunction(
  () => /SAME PHYSICS|DIFFERENT|FAILED/.test(document.body.innerText), { timeout: 180000 });
const text = await page.evaluate(() => document.body.innerText);
await browser.close(); server.close();

console.log(text);
// A favicon 404 is the browser asking, not the page failing; anything else the
// page itself asked for and did not get.
const real = missing.filter(p => p !== '/favicon.ico');
if (real.length) console.log(`\nMISSING ASSETS: ${real.join(', ')}`);
if (errors.length) console.log(`\nPAGE ERRORS: ${errors.join(' | ')}`);
const ok = /SAME PHYSICS/.test(text) && !real.length && !errors.length;
console.log(ok ? '\nPAGECHECK OK — the probe runs and reproduces the desk trajectory in Chromium'
               : '\nPAGECHECK FAILED');
if (!ok) process.exitCode = 1;
