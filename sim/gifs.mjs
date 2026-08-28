// Demo GIFs: each intent recorded from the running simulator, staged so it has
// the conditions it actually needs. A wall flip filmed in the middle of an open
// floor shows nothing; a riser push with no stair shows less.
//
// Frames come from the canvas at 20 fps and ffmpeg assembles them with a
// generated palette — a default 256-colour quantise turns the duck's grey
// shells into banded mud.
import puppeteer from 'puppeteer-core';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';

const URL_ = process.argv[2] || 'http://127.0.0.1:8099/';
const OUT = 'gifs';
const FPS = 20;

// key, name, seconds to film, and how to stage it.
const SHOTS = [
  { key: 'q', id: 'kick-left',   secs: 2.5, stage: {} },
  { key: 'f', id: 'pick-up',     secs: 4.5, stage: {} },
  { key: 'c', id: 'sit',         secs: 3.5, stage: {} },
  { key: 'v', id: 'stand-up',    secs: 4.0, stage: { pre: 'c', preWait: 3000 } },
  { key: 'x', id: 'forward-roll', secs: 4.0, stage: {} },
  { key: 'b', id: 'back-roll',   secs: 4.0, stage: {} },
  { key: 't', id: 'wall-flip',   secs: 4.0, stage: { nearWall: true } },
  { key: 'g', id: 'step-up',     secs: 4.5, stage: { rise: 26, atStep: true } },
  { key: 'h', id: 'lever-up',    secs: 5.0, stage: { rise: 40, atStep: true } },
  { key: 'y', id: 'riser-up',    secs: 5.5, stage: { rise: 55, atStep: true } },
  { key: null, id: 'walking',    secs: 4.0, stage: { hold: 'ArrowUp' } },
  { key: null, id: 'skating',    secs: 4.0, stage: { variant: 'rollers', hold: 'ArrowUp' } },
];

const browser = await puppeteer.launch({ executablePath: '/usr/bin/chromium', headless: 'new',
  args: ['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 760, height: 520, deviceScaleFactor: 1 });
await page.goto(URL_ + '?demo=1&v=' + Date.now(), { waitUntil: 'domcontentloaded', timeout: 120000 });
await page.waitForFunction(() => document.body.classList.contains('ready'), { timeout: 180000 });
const wait = ms => new Promise(r => setTimeout(r, ms));

async function setRise(mm) {
  await page.evaluate(v => {
    const el = document.getElementById('rise');
    el.value = v; el.dispatchEvent(new Event('input', { bubbles: true }));
    const c = document.getElementById('count');
    c.value = 3; c.dispatchEvent(new Event('input', { bubbles: true }));
  }, mm);
}
async function setVariant(v) {
  await page.select('#variant', v);
  await page.waitForFunction(() => document.getElementById('status').textContent === '', { timeout: 180000 });
  await wait(500);
}

let made = 0;
for (const shot of SHOTS) {
  const dir = `${OUT}/${shot.id}`;
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });

  if (shot.stage.variant) await setVariant(shot.stage.variant);
  else await setVariant('legs');
  await setRise(shot.stage.rise || 0);
  await page.keyboard.press('r');
  await wait(700);

  if (shot.stage.nearWall) {
    // The flip was searched at a specific distance from the wall — 0.065 m —
    // and walking the duck up to it approximately is not the same thing.
    // Measured: staged by walking, it does not flip at all.
    // Settle under the STAND policy, as the search did. The page otherwise
    // settles under the walking policy, and the flip starts from a different
    // state and does not fire.
    await page.keyboard.press('z');
    await wait(900);
    await page.evaluate(g => window.__demo.place(1.5 - 0.05 - g), 0.0654);
    await wait(700);
  }
  if (shot.stage.atStep) {
    // Likewise the riser push: it starts 0.127 m off the riser face.
    await page.evaluate(g => window.__demo.place(0.12 - 0.07 - g), 0.1266);
    await wait(700);
  }
  if (shot.stage.pre) { await page.keyboard.press(shot.stage.pre); await wait(shot.stage.preWait || 2500); }

  const el = await page.$('#view');
  if (shot.stage.hold) await page.keyboard.down(shot.stage.hold);
  if (shot.key) await page.keyboard.press(shot.key);

  const frames = Math.round(shot.secs * FPS);
  for (let i = 0; i < frames; i++) {
    await el.screenshot({ path: `${dir}/f${String(i).padStart(3, '0')}.png` });
  }
  if (shot.stage.hold) await page.keyboard.up(shot.stage.hold);

  const gif = `${OUT}/${shot.id}.gif`;
  const pal = `${dir}/pal.png`;
  spawnSync('ffmpeg', ['-y','-v','error','-framerate',String(FPS),'-i',`${dir}/f%03d.png`,
    '-vf','fps=20,scale=640:-1:flags=lanczos,palettegen=stats_mode=diff', pal]);
  spawnSync('ffmpeg', ['-y','-v','error','-framerate',String(FPS),'-i',`${dir}/f%03d.png`,'-i',pal,
    '-lavfi','fps=20,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3', gif]);
  const kb = fs.existsSync(gif) ? (fs.statSync(gif).size/1024).toFixed(0) : '—';
  console.log(`GIF ${shot.id.padEnd(14)} ${frames} frames  ${kb} KB`);
  fs.rmSync(dir, { recursive: true, force: true });
  made++;
}
console.log(`made ${made} gifs`);
await browser.close();
