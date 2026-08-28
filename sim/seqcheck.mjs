// Does the sequence machinery actually work, and does the approach converge?
//
// The convergence number is the one that matters. Every keyframe move was
// searched from a duck placed at an exact mark; a duck that WALKED there is
// only as good as this controller, and the page's own copy says the duck
// drifts, will not walk slowly, and covers half the ground it is asked to.
import puppeteer from 'puppeteer-core';
const URL_ = (process.argv[2] || 'http://127.0.0.1:8099/') + '?demo=1&v=' + Date.now();
const browser = await puppeteer.launch({ executablePath:'/usr/bin/chromium', headless:'new',
  args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
const errs = [];
page.on('pageerror', e => errs.push(e.message));
await page.setViewport({width:1600,height:1000});
await page.goto(URL_, {waitUntil:'domcontentloaded'});
await page.waitForFunction(()=>document.body.classList.contains('ready'),{timeout:180000});
await new Promise(r=>setTimeout(r,900));
const ok=[], bad=[];
const check=(n,p,d='')=>{ (p?ok:bad).push(n); console.log(`${p?'PASS':'FAIL'}  ${n}${d?'  — '+d:''}`); };

// ── 1. the bugs that made this untestable ─────────────────────────────────
const dis = () => page.$$eval('.keys button', b=>b.filter(x=>x.disabled).length);
await page.keyboard.press('q');
await new Promise(r=>setTimeout(r,4500));
check('buttons come back after a kick', await dis()===0, `${await dis()} still disabled`);
await page.keyboard.press('r'); await new Promise(r=>setTimeout(r,500));

// ── 2. readiness ───────────────────────────────────────────────────────────
const flat = await page.evaluate(()=>window.__demo.seq.readyFor('lever_up'));
check('a stair move is not ready on a flat floor', flat && !flat.ok && flat.fix==='stairs', flat && flat.reason);
const roll = await page.evaluate(()=>window.__demo.seq.readyFor('back_roll'));
check('a move that needs nothing reports nothing', roll===null);
const tips = await page.$$eval('.keys button', b=>b.map(x=>x.title));
check('no tooltip says "undefined"', !tips.some(t=>/undefined/.test(t)), tips[0]);
check('a stair move says what it needs', tips.some(t=>/10 mm step/.test(t)),
      tips.find(t=>/10 mm step/.test(t)));

// ── 3. one move lights one button ─────────────────────────────────────────
await page.evaluate(()=>{const e=document.getElementById('rise');e.value=10;e.dispatchEvent(new Event('input',{bubbles:true}));});
await new Promise(r=>setTimeout(r,300));
await page.evaluate(()=>window.__demo.seq.runner.play([{kind:'place',x:0.12-0.07-0.1266,y:1.305},{kind:'settle',ticks:25,approach:0.078},{kind:'move',id:'riser_up'}]));
await page.waitForFunction(()=>document.querySelectorAll('.keys button.on').length>0,{timeout:30000}).catch(()=>{});
const lit = await page.$$eval('.keys button.on', b=>b.map(x=>x.textContent.trim()));
check('one running move lights exactly one button', lit.length===1, JSON.stringify(lit));
await page.waitForFunction(()=>window.__demo.seq.runner.state!=='running',{timeout:60000}).catch(()=>{});

// ── 4. does walking actually get there? ───────────────────────────────────
console.log('\n  approach convergence, walking from four starts:');
const runs=[];
for (const [sx,sy] of [[-0.6,1.30],[-0.4,0.9],[0.3,1.7],[-0.8,1.6]]) {
  const r = await page.evaluate(async ([sx,sy])=>{
    const S=window.__demo.seq;
    window.__demo.place(sx,sy);
    const t={x:0.12-0.07-0.1266, y:1.305, yaw:0};
    await S.runner.play([{kind:'goto',...t}]);
    const t0=performance.now();
    while (S.runner.state==='running' && performance.now()-t0<70000) await new Promise(r=>setTimeout(r,120));
    const p=S.pose();
    return { d:Math.hypot(t.x-p.x,t.y-p.y), yaw:Math.abs(Math.atan2(Math.sin(-p.yaw),Math.cos(-p.yaw))),
             state:S.runner.state, note:S.runner.note };
  },[sx,sy]);
  runs.push(r);
  console.log(`    from ${sx},${sy}  ->  ${Math.round(r.d*1000)} mm, ${Math.round(r.yaw*57.3)}deg   ${r.state}  ${r.note}`);
}
const arrived = runs.filter(r=>r.d<=0.030 && r.yaw<=0.14).length;
check(`walking reaches the mark`, arrived===runs.length, `${arrived}/${runs.length} inside 30 mm and 8 deg`);

// ── 5. record and replay ──────────────────────────────────────────────────
const rt = await page.evaluate(async ()=>{
  const S=window.__demo.seq;
  S.setProgram([{kind:'stairs',rise:0.010,count:4,run:0.28},
                {kind:'place',x:-0.0766,y:1.305},
                {kind:'settle',ticks:25,approach:0.078},
                {kind:'move',id:'riser_up'}]);
  await S.runner.play();
  const t0=performance.now(); const seen=new Set();
  while (S.runner.state==='running' && performance.now()-t0<60000) { seen.add(S.runner.pc); await new Promise(r=>setTimeout(r,60)); }
  return { steps:[...seen].sort(), stairs:S.stairs(), state:S.runner.state };
});
check('a recorded sequence runs every step', rt.steps.length===4, JSON.stringify(rt.steps));
check('the sequence set the staircase', Math.abs(rt.stairs.rise-0.010)<1e-6, JSON.stringify(rt.stairs));
check('no page errors', errs.length===0, errs.slice(0,2).join(' | '));
console.log(`\n${ok.length}/${ok.length+bad.length} passed`);
await browser.close();
process.exit(bad.length?1:0);
