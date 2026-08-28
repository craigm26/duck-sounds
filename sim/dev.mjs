// The fast loop: serve site/ locally and drive it in headless Chromium.
//
// Deploying to Cloudflare to see whether a change worked costs 30-60 s and
// several confusing failure modes of its own (stale caches, a stale asset that
// was never re-copied). This does the same check against a local server in a
// few seconds, and it is the same browsercheck the deployed site gets, so a
// pass here means the same thing.
//
//   node dev.mjs          serve, check once, report
//   node dev.mjs --watch  re-check whenever a file under site/ changes
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const PORT = 8099;
const SITE = path.resolve('../site');
const watch = process.argv.includes('--watch');

const server = spawn('node', ['serve.mjs', SITE, String(PORT)], { stdio: 'inherit' });
process.on('exit', () => server.kill());
process.on('SIGINT', () => { server.kill(); process.exit(0); });

const check = () => new Promise(res => {
  const p = spawn('node', ['browsercheck.mjs', `http://127.0.0.1:${PORT}/`], { stdio: 'inherit' });
  p.on('exit', res);
});

await new Promise(r => setTimeout(r, 1200));
await check();

if (watch) {
  let timer = null;
  fs.watch(SITE, { recursive: true }, (_e, file) => {
    if (!file || file.endsWith('.png')) return;
    clearTimeout(timer);
    timer = setTimeout(async () => {
      console.log(`\n── ${file} changed ──`);
      await check();
    }, 400);
  });
  console.log('watching site/ — edit and it re-checks');
} else {
  server.kill();
}
