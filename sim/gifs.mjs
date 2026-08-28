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
const NEARGAP = 0.0654;   // the distance the wall flip was searched at
const STEPGAP = 0.1266;   // and the riser push

// key, name, seconds to film, and how to stage it.
const SHOTS = [
  { key: 'q', id: 'kick-left',   secs: 2.5, stage: {} },
  { key: 'f', id: 'pick-up',     secs: 4.5, stage: {} },
  { key: 'c', id: 'sit',         secs: 3.5, stage: {} },
  { key: 'v', id: 'stand-up',    secs: 4.0, stage: { pre: 'c', preWait: 3000 } },
  { key: 'x', id: 'forward-roll', secs: 4.0, stage: {} },
  { key: 'b', id: 'back-roll',   secs: 4.0, stage: {} },
  { key: 'g', id: 'step-up',     secs: 4.5, stage: { rise: 10, atStep: true } },
  { key: 'h', id: 'lever-up',    secs: 5.0, stage: { rise: 10, atStep: true } },
  { key: 'y', id: 'riser-up',    secs: 5.5, stage: { rise: 10, atStep: true } },
  { key: 'u', id: 'climb',       secs: 6.0, stage: { rise: 10, atStep: true } },
  { key: 't', id: 'wall-flip',   secs: 4.0, stage: { nearWall: true } },
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
    // Order matters and I had it backwards. The search does: position, THEN
    // settle under the stand policy, THEN play the move. Settling first and
    // placing afterwards throws the settle away, because place() resets the
    // pose and the action history — which is the whole point of it.
    await page.evaluate(g => window.__demo.place(1.5 - 0.05 - g), NEARGAP);
    // 25 ticks under the stand policy holding the approach command — the exact
    // warm-up the search ran. Not a wall-clock wait under some other policy:
    // that leaves the pose right and the joint velocities wrong.
    await page.evaluate(a => window.__demo.settle(25, a), 0.07513);
    await wait(150);
  }
  if (shot.stage.atStep) {
    // The page lays its stairs out from x = 0.45; the search used 0.12. Place
    // the duck relative to where the stairs ACTUALLY are, or it stands beside
    // them and the recording shows a duck ignoring a staircase.
    await page.evaluate(g => window.__demo.place(0.45 - 0.07 - g, 1.305), STEPGAP);
    await page.evaluate(a => window.__demo.settle(25, a), shot.stage.approach || 0);
    await wait(150);
  }
  if (shot.stage.pre) { await page.keyboard.press(shot.stage.pre); await wait(shot.stage.preWait || 2500); }

  const el = await page.$('#view');
  if (shot.stage.hold) await page.keyboard.down(shot.stage.hold);
  if (shot.key) await page.keyboard.press(shot.key);

  // Capture against the SIMULATION clock, not the wall clock.
  //
  // Screenshotting blocks requestAnimationFrame, which is what drives the
  // physics — so taking 80 shots back to back advances the sim by almost
  // nothing and the GIF shows a duck standing still while, in the page, it has
  // done the whole move. Measured that way the wall flip reached 177 degrees
  // and the recording of it showed a lean. So: wait for the tick counter to
  // advance, then take a frame.
  const ticksTotal = Math.round(shot.secs * 50);
  const frames = Math.round(shot.secs * FPS);
  const perFrame = Math.max(1, Math.round(ticksTotal / frames));
  const tickNow = () => page.$eval('#hud', el => {
    const m = el.textContent.match(/tick (\d+)/); return m ? +m[1] : 0;
  });
  const t0 = await tickNow();
  for (let i = 0; i < frames; i++) {
    const want = t0 + i * perFrame;
    for (let guard = 0; guard < 60 && (await tickNow()) < want; guard++) await wait(20);
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
