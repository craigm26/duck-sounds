// A local static server for site/, so the edit->check loop does not go through
// a Cloudflare deploy. Correct MIME types matter: .wasm must be application/wasm
// or the browser refuses to instantiate it, and .mjb/.bin must not be sniffed.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(process.argv[2] || '../site');
const PORT = +(process.argv[3] || 8099);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'application/javascript',
  '.mjs': 'application/javascript', '.json': 'application/json',
  '.wasm': 'application/wasm', '.xml': 'application/xml',
  '.mjb': 'application/octet-stream', '.bin': 'application/octet-stream',
  '.stl': 'model/stl', '.css': 'text/css',
};

http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  let file = path.join(ROOT, url === '/' ? 'index.html' : url);
  if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404).end('not found: ' + url); return; }
    res.writeHead(200, {
      'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(buf);
  });
}).listen(PORT, '127.0.0.1', () => console.log(`SERVING ${ROOT} on http://127.0.0.1:${PORT}`));
