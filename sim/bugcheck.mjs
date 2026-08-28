// Verify the audit's claims against the live page rather than trusting them.
import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/');
const browser = await puppeteer.launch({ executablePath:'/usr/bin/chromium', headless:'new',
  args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({width:1600,height:1000});
await page.goto(URL_+'?v='+Date.now(),{waitUntil:'domcontentloaded'});
await page.waitForFunction(()=>document.body.classList.contains('ready'),{timeout:180000});
await new Promise(r=>setTimeout(r,800));

const dis = () => page.$$eval('.keys button', b => b.filter(x=>x.disabled).length);
const on  = () => page.$$eval('.keys button', b => b.filter(x=>x.classList.contains('on')).map(x=>x.textContent.trim()));

console.log('CLAIM 1  kick permanently disables all 13 buttons');
console.log('  before          disabled', await dis());
await page.keyboard.press('q');
await new Promise(r=>setTimeout(r,400));
console.log('  0.4s after kick disabled', await dis());
await new Promise(r=>setTimeout(r,4000));
console.log('  4.4s after kick disabled', await dis(), '  <-- should be 0');
await page.click('#reset');
await new Promise(r=>setTimeout(r,600));
console.log('  after reset     disabled', await dis(), '  <-- should be 0');

console.log('\nCLAIM 2  one keyframe move lights several buttons');
await page.evaluate(()=>{const e=document.getElementById('rise');e.value=10;e.dispatchEvent(new Event('input',{bubbles:true}));});
await page.reload({waitUntil:'domcontentloaded'});
await page.waitForFunction(()=>document.body.classList.contains('ready'),{timeout:180000});
await new Promise(r=>setTimeout(r,800));
await page.keyboard.press('y');                       // riser_up, a track move
await new Promise(r=>setTimeout(r,500));
console.log('  lit while riser_up runs:', JSON.stringify(await on()), ' <-- should be exactly one');

console.log('\nCLAIM 3  move tooltips say "undefined"');
console.log(' ', JSON.stringify(await page.$$eval('.keys button', b=>b.map(x=>x.title).slice(0,4))));

console.log('\nCLAIM 4  --head-h:auto collapses the rig without JS');
const noJs = await browser.newPage();
await noJs.setJavaScriptEnabled(false);
await noJs.setViewport({width:1600,height:1000});
await noJs.goto(URL_+'?v='+Date.now(),{waitUntil:'domcontentloaded'});
await new Promise(r=>setTimeout(r,600));
console.log('  rig height with JS off:', await noJs.$eval('#rig', e=>Math.round(e.getBoundingClientRect().height)), 'px  <-- should be a full stage');
await noJs.close();

console.log('\nCLAIM 5  AR button unreachable on a touch device');
const touch = await browser.newPage();
await touch.setViewport({width:1180,height:820,hasTouch:true});
await touch.goto(URL_+'?v='+Date.now(),{waitUntil:'domcontentloaded'});
await touch.waitForFunction(()=>document.body.classList.contains('ready'),{timeout:180000});
console.log('  is-touch:', await touch.evaluate(()=>document.body.classList.contains('is-touch')),
            ' .dock-drive display:', await touch.$eval('.dock-drive', e=>getComputedStyle(e).display),
            ' <-- #xr lives inside it');
console.log('\nCLAIM 6  focusable buttons inside aria-hidden');
console.log('  .touch-layer aria-hidden:', await touch.$eval('#touch', e=>e.getAttribute('aria-hidden')),
            ' focusable inside:', await touch.$$eval('#touch button', b=>b.length));
await touch.close();

console.log('\nCLAIM 7  servo sliders have no accessible name');
const names = await page.$$eval('#servoList input', els => els.slice(0,3).map(e => ({id:e.id||null, label:e.labels?e.labels.length:0, aria:e.getAttribute('aria-label')})));
console.log(' ', JSON.stringify(names));
await browser.close();
