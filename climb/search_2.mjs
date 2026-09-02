// Strategy B, second leg — HEAD PRESS + TRUNK TWIST, with the two levers the
// first leg (climb/search_1.mjs) did not have.
//
// WHAT THE FIRST LEG MEASURED, AND WHY THIS FILE EXISTS
// climb/search_1-results_*.json: at every rise the best of 400-600 evaluations
// ends the episode UPRIGHT ON THE FLOOR, trunk z ~= 116 mm (a standing duck),
// trunk x 9 mm, trunk peak x 82 mm. The criterion needs trunk x > 0.12 m
// (sim/climb_lib.mjs:150). Head contact fraction peaked at 0.10-0.18.
// So the twist never got a chance: the head barely reached the tread and the
// trunk never travelled the ~108 mm it must travel.
//
// climb/headreach-results.json is the reason. With the head extended and held
// and NOTHING else moving, from gap=0.02 (start x = 30 mm) the trunk ends at
// x = -37 to -44 mm at every rise: reaching the head out pushes the duck
// BACKWARDS off the step. The stand policy holds station against a zero
// velocity command, so the head-reach reaction has nowhere to go but the
// trunk. A press that starts 90 mm short of the riser cannot land.
//
// TWO CHANGES, both deviations from the strategy brief, both reported:
//   1. `approach` — the policy's vx command (site/duckloop.mjs:57) — is
//      SEARCHED over [0, 0.45] instead of the brief's fixed 0. This is the
//      only actuator in the episode that can translate the trunk forward
//      against the stand policy's station-keeping.
//   2. `gap` is searched over [0.0, 0.035] instead of [0.020, 0.045]: start
//      the duck's front geom touching the riser, because the jaw's forward
//      reach (~87 mm ahead of the trunk, climb/reach-max.mjs) minus the
//      brief's minimum gap leaves the jaw short of the tread's x = 0.12 edge.
// Everything else — the 6-frame skeleton, the joint bounds, the twistDir
// binary, the side/twistDir coupling — is the brief's, unchanged.
//
// SEARCH: cross-entropy method (population 14, elite 4) rather than the first
// leg's restart hill climbing. 45 correlated parameters and ~1000 evaluations
// is the regime where a population mean beats a single walker. Warm-started
// from the first leg's per-rise best when climb/search_1-results_*.json has one.
//
// Run from sim/ (rig2/climb_lib read scene.mjb, duckkit-constants.json and the
// ONNX by cwd-relative path):
//   cd ~/projects/duck-sounds/sim && RISE=40 SECONDS=500 SEED=101 node ../climb/search_2.mjs
//
// Env: RISE (mm, comma list), SECONDS (wall budget per rise), SEED, TWIST
// (+1/-1/both), TAG (output suffix), WARM (0 to disable the warm start).
import fs from 'node:fs';
import { run, HOME, LO, HI } from './rig2.mjs';

const J = { lhy:0, lhr:1, lhp:2, lk:3, la:4, np:5, hp:6, hy:7, hr:8,
            rhy:9, rhr:10, rhp:11, rk:12, ra:13 };

// ── the searched box (the brief's, plus approach; gap widened downward) ─────
const B = {
  t1:[0.25,0.55], d2:[0.15,0.40], d3:[0.15,0.45], d4:[0.15,0.45], d5:[0.15,0.50], d6:[0.25,0.70],
  hp1:[0.70,1.40], k1:[-1.20,0.60], a1:[-1.20,0.90], hy1:[0.10,0.4363],
  neck1:[0.00,0.35], head1:[0.20,0.60],
  neck2:[-1.5708,-1.20], head2:[0.40,1.10], roll2:[-0.4363,0.4363],
  yaw3:[0.60,2.20], hy3:[0.0,0.4363], hr3:[-0.30,0.38],
  neck3:[-1.5708,-1.10], head3:[0.40,1.20],
  inHip4:[1.10,1.5708], inKnee4:[-1.30,-0.55], inAnk4:[0.85,1.45], inRoll4:[-0.384,0.384],
  outHip5:[-1.5708,1.5708], outKnee5:[-1.5708,1.5708], outAnk5:[-1.00,1.00], outRoll5:[-0.384,0.384],
  yaw5:[0.30,2.20], neck5:[-1.5708,-0.80], head5:[0.20,1.20],
  yaw6:[-0.40,0.40], neck6:[0.40,1.0472], head6:[-1.5708,-0.60], roll6:[-0.4363,0.4363],
  hy6:[-0.20,0.20], hp6:[-1.40,1.40], k6:[-1.40,1.40], a6:[-1.00,1.00],
  blend:[0.8,2.4], gap:[0.0,0.035], side:[0.0,0.085],
  approach:[0.0,0.45],                 // <- the new lever
};
const KEYS = Object.keys(B);
const clampJ = pose => pose.map((v,k)=>Math.min(Math.max(v,LO[k]),HI[k]));

/** The brief's 6 searched frames + a HOME tail. Identical to search_1.mjs. */
function trackOf(p, dir){
  const F1 = HOME.slice();
  F1[J.lhp] = -p.hp1;            F1[J.rhp] = +p.hp1;
  F1[J.lk]  = HOME[J.lk]+p.k1;   F1[J.rk]  = HOME[J.rk]-p.k1;
  F1[J.la]  = HOME[J.la]+p.a1;   F1[J.ra]  = HOME[J.ra]-p.a1;
  F1[J.lhy] = dir*p.hy1;         F1[J.rhy] = -dir*p.hy1;
  F1[J.np]  = p.neck1;           F1[J.hp]  = p.head1;
  const F2 = F1.slice();
  F2[J.np]=p.neck2; F2[J.hp]=p.head2; F2[J.hr]=p.roll2;
  const F3 = F2.slice();
  F3[J.hy]=dir*p.yaw3; F3[J.lhy]=-dir*p.hy3; F3[J.rhy]=dir*p.hy3;
  F3[J.lhr]=p.hr3; F3[J.rhr]=-p.hr3; F3[J.np]=p.neck3; F3[J.hp]=p.head3;
  const F4 = F3.slice();                       // inboard = left
  F4[J.lhp]=p.inHip4; F4[J.lk]=p.inKnee4; F4[J.la]=p.inAnk4; F4[J.lhr]=p.inRoll4;
  const F5 = F4.slice();                       // outboard = right
  F5[J.rhp]=p.outHip5; F5[J.rk]=p.outKnee5; F5[J.ra]=p.outAnk5; F5[J.rhr]=p.outRoll5;
  F5[J.hy]=dir*p.yaw5; F5[J.np]=p.neck5; F5[J.hp]=p.head5;
  const F6 = F5.slice();
  F6[J.hy]=p.yaw6; F6[J.np]=p.neck6; F6[J.hp]=p.head6; F6[J.hr]=p.roll6;
  F6[J.lhy]=p.hy6; F6[J.rhy]=-p.hy6;
  F6[J.lhp]=HOME[J.lhp]+p.hp6; F6[J.rhp]=HOME[J.rhp]-p.hp6;
  F6[J.lk]=HOME[J.lk]+p.k6;    F6[J.rk]=HOME[J.rk]-p.k6;
  F6[J.la]=HOME[J.la]+p.a6;    F6[J.ra]=HOME[J.ra]-p.a6;
  const t1=p.t1, t2=t1+p.d2, t3=t2+p.d3, t4=t3+p.d4, t5=t4+p.d5, t6=t5+p.d6;
  return [{t:t1,pose:clampJ(F1)},{t:t2,pose:clampJ(F2)},{t:t3,pose:clampJ(F3)},
          {t:t4,pose:clampJ(F4)},{t:t5,pose:clampJ(F5)},{t:t6,pose:clampJ(F6)},
          {t:t6+0.7,pose:HOME.slice()}];
}

const cl=(v,a,b)=>Math.max(a,Math.min(b,v));
/**
 * The brief's reward, plus three TERMINAL terms. The first leg's reward was
 * entirely max-over-ticks: a duck that lunged and fell back scored the same as
 * one that arrived and stayed, and the criterion is a terminal test held for
 * 1 s. feetUpMax is the criterion's own feetUp test evaluated per tick
 * (climb/rig2.mjs) — the most direct gradient toward it that exists.
 */
function reward(r, h){
  return 3.0*cl((r.maxX - r.x0)/0.20,0,1)
    + 3.0*cl((r.maxZ - h)/0.12,0,1)
    + 2.0*r.headFrac
    + 2.0*r.footOver
    + 1.0*cl(r.yawPlanted/0.9,0,1)*r.headFrac
    + 1.0*r.upFrac
    - 5.0*Math.max(0, r.x0 - r.minX)/0.10
    - 2.0*Math.max(0, r.maxY - 1.44)/0.01
    - 0.5*r.satFrac
    // terminal shaping (new)
    + 3.0*cl((r.x - r.x0)/(0.12 - r.x0 + 1e-9),0,1)     // where it ENDED, vs the criterion's x
    + 2.0*cl((r.above - 0.0)/0.095,0,1)                  // terminal trunk height over the tread
    + 2.0*(r.feetUpMax/2)                                // criterion's feetUp, graded per tick
    + 1.0*r.bothFootOver                                 // BOTH soles toward the tread, not just the outboard
    + (r.onTop?100:0);
}

// ── deterministic RNG ───────────────────────────────────────────────────────
let S = (+process.env.SEED || 777)>>>0;
const rnd = () => { S ^= S<<13; S>>>=0; S ^= S>>>17; S ^= S<<5; S>>>=0; return S/4294967296; };
const gauss = () => { let u=0,v=0; while(!u)u=rnd(); while(!v)v=rnd();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
const sideBox = dir => dir>0 ? [0.045,0.085] : [0.0,0.030];

function randP(dir){
  const p={}; for(const k of KEYS){ const [a,b]=B[k]; p[k]=a+rnd()*(b-a); }
  const [sa,sb]=sideBox(dir); p.side = sa+rnd()*(sb-sa);
  return p;
}
/** Sample from a CEM Gaussian (per-parameter mean + sigma, both in raw units). */
function sample(mu, sig, dir){
  const p={}; for(const k of KEYS){ const [a,b]=B[k]; p[k]=cl(mu[k]+gauss()*sig[k],a,b); }
  const [sa,sb]=sideBox(dir); p.side=cl(p.side,sa,sb);
  return p;
}

const RISES = (process.env.RISE||'40').split(',').map(s=>+s);
const SECONDS = +(process.env.SECONDS||400);
const TWIST = process.env.TWIST||'both';
const TAG = process.env.TAG||'';
const WARM = process.env.WARM!=='0';
const DIRS = TWIST==='both'?[+1,-1]:[+TWIST];
const POP = 14, ELITE = 4;

/** The first leg's best parameters for this rise, if it left any. */
function warmStart(rmm){
  if (!WARM) return null;
  const files = fs.readdirSync('../climb');
  // prefer leg 2's own best for this rise (WARM2), then leg 1's
  for (const f of files){
    if (!/^search_2-results_[ab]\d\.json$/.test(f)) continue;
    if (process.env.WARM2!=='1') continue;
    try { const d=JSON.parse(fs.readFileSync('../climb/'+f,'utf8'));
      if (d[rmm] && d[rmm].params) return { p:{...d[rmm].params}, dir:d[rmm].twistDir, src:f }; } catch {}
  }
  for (const f of files){
    if (!/^search_1-results.*\.json$/.test(f)) continue;
    try {
      const d = JSON.parse(fs.readFileSync('../climb/'+f,'utf8'));
      if (d[rmm] && d[rmm].params) return { p:{...d[rmm].params}, dir:d[rmm].twistDir, src:f };
    } catch {}
  }
  return null;
}

// ROBUST=1: a two-stage objective. Every candidate is scored at the nominal
// start; only a candidate within 10% of the incumbent is re-run at the -10 and
// +10 mm start offsets, and then its score is the MINIMUM over the three. The
// first pass at 40 mm found an 11.2 that scored -551 mm terminal x at +10 mm —
// a knife-edge, not a move. Scoring the worst case is the only way the search
// can see that. Two-stage because the full three-start score costs 3x and 95%
// of candidates never come near the incumbent.
const ROBUST = process.env.ROBUST === '1';
const OFFS = [-0.010, 0.010];
const scoreOnce = async (p,dir,h,gapOff=0) => {
  const tr = trackOf(p,dir);
  const r = await run(tr, {blend:p.blend, gap:Math.max(0,p.gap+gapOff), side:p.side,
                           approach:p.approach}, h);
  return { r, score: reward(r,h), tr };
};
let incumbent = -1e9;
const evalP = async (p,dir,h) => {
  const c = await scoreOnce(p,dir,h);
  if (!ROBUST || c.score < incumbent*0.90) return c;
  let worst = c.score;
  for (const d of OFFS){ const o = await scoreOnce(p,dir,h,d); if (o.score < worst) worst = o.score; }
  return { r:c.r, tr:c.tr, score:worst, nominal:c.score };
};
const line = (rmm,e,tag,c) => `[${rmm}mm] e${e} ${tag} ${c.score.toFixed(3)} dir${c.dir>0?'+':'-'} `
  + `onTop=${c.r.onTop} x=${(c.r.x*1000).toFixed(0)} z=${(c.r.z*1000).toFixed(0)} peakX=${(c.r.maxX*1000).toFixed(0)} `
  + `peakZ=${(c.r.maxZ*1000).toFixed(0)} head=${c.r.headFrac.toFixed(2)} riser=${c.r.riserFrac.toFixed(2)} `
  + `foot=${c.r.footOver.toFixed(2)} feetUpMax=${c.r.feetUpMax} yaw=${c.r.yawPlanted.toFixed(2)} `
  + `vx=${c.p.approach.toFixed(2)} gap=${(c.p.gap*1000).toFixed(0)}`;

const out = {};
for (const rmm of RISES){
  const h = rmm/1000;
  const t0 = Date.now();
  const left = () => SECONDS - (Date.now()-t0)/1000;
  let best = null, evals = 0; const log = [];
  const note = c => { best={...c}; incumbent=c.score; log.push({evals,score:+c.score.toFixed(3),dir:c.dir,
      params:{...c.p}, r:{onTop:c.r.onTop,x_mm:+(c.r.x*1000).toFixed(1),z_mm:+(c.r.z*1000).toFixed(1),
      peakX_mm:+(c.r.maxX*1000).toFixed(1),peakZ_mm:+(c.r.maxZ*1000).toFixed(1),
      headFrac:+c.r.headFrac.toFixed(3),riserFrac:+c.r.riserFrac.toFixed(3),
      feetUpMax:c.r.feetUpMax,feetUp:c.r.feetUp,yawPlanted:+c.r.yawPlanted.toFixed(3)}});
    console.log(line(rmm,evals,'NEW',c)); };

  // seed round: the warm start, plus randoms, per twist direction
  const seeds = [];
  const ws = warmStart(rmm);
  if (ws){ ws.p.approach = 0; ws.p.gap = cl(ws.p.gap,0,0.035); seeds.push({p:ws.p,dir:ws.dir});
    // the same first-leg pose, but WITH forward drive — the one-variable test
    if (process.env.WARM2!=='1') for (const vx of [0.12,0.25,0.40]) seeds.push({p:{...ws.p,approach:vx},dir:ws.dir});
    console.log(`[${rmm}mm] warm start from ${ws.src} (dir ${ws.dir}) + vx probes`);
  }
  for (const dir of DIRS) for (let i=0;i<6;i++) seeds.push({p:randP(dir),dir});
  for (const s of seeds){
    if (left()<5) break;
    const c = await evalP(s.p,s.dir,h); evals++;
    const cc = {...c,p:s.p,dir:s.dir};
    if (!best || c.score>best.score) note(cc);
  }

  // CEM per direction, alternating generations, each with its own Gaussian
  const G = {};
  for (const dir of DIRS){
    const mu={}, sig={};
    const base = (best && best.dir===dir) ? best.p : randP(dir);
    for (const k of KEYS){ const [a,b]=B[k]; mu[k]=base[k]; sig[k]=(b-a)*0.30; }
    G[dir]={mu,sig};
  }
  let gen=0;
  while (left() > 8){
    const dir = DIRS[gen % DIRS.length]; gen++;
    const {mu,sig} = G[dir];
    const pool=[];
    for (let i=0;i<POP && left()>5;i++){
      const p = (i===0 && best && best.dir===dir) ? {...best.p} : sample(mu,sig,dir);
      const c = await evalP(p,dir,h); evals++;
      const cc={...c,p,dir}; pool.push(cc);
      if (!best || c.score>best.score) note(cc);
    }
    if (pool.length<ELITE) break;
    pool.sort((a,b)=>b.score-a.score);
    const el = pool.slice(0,ELITE);
    for (const k of KEYS){
      const m = el.reduce((s,e)=>s+e.p[k],0)/el.length;
      let v = el.reduce((s,e)=>s+(e.p[k]-m)**2,0)/el.length;
      const [a,b]=B[k];
      // 0.20 of the box is the sigma floor: a CEM that collapses on 45
      // parameters in ~70 generations has stopped searching, and the elite
      // here are 4 samples, so the empirical variance is badly underestimated.
      mu[k]=m; sig[k]=Math.max(Math.sqrt(v)*1.3, (b-a)*0.05);
      sig[k]=Math.min(sig[k],(b-a)*0.45);
    }
  }

  // ── robustness: the criterion re-run at start offsets -10 / 0 / +10 mm ────
  // The start x is 0.12 - 0.07 - gap (sim/climb_lib.mjs:113), so shifting gap
  // by -/+10 mm shifts the start by +/-10 mm. gap is floored at 0 (a negative
  // gap would spawn the duck inside the riser).
  const checks=[];
  for (const d of [-0.010,0,0.010]){
    const g = best.p.gap + d;
    const r = await run(best.tr, {blend:best.p.blend, gap:Math.max(0,g), side:best.p.side,
                                  approach:best.p.approach}, h);
    checks.push({startOffset_mm:+(-d*1000).toFixed(0), gap_mm:+(Math.max(0,g)*1000).toFixed(1),
      onTop:r.onTop, x_mm:+(r.x*1000).toFixed(1), z_mm:+(r.z*1000).toFixed(1),
      above_mm:+(r.above*1000).toFixed(1), feetUp:r.feetUp, up:r.up,
      peakZ_mm:+(r.maxZ*1000).toFixed(1), peakX_mm:+(r.maxX*1000).toFixed(1),
      headFrac:+r.headFrac.toFixed(3), riserFrac:+r.riserFrac.toFixed(3),
      feetUpMax:r.feetUpMax});
  }
  const cleared = checks.filter(c=>c.onTop).length;
  const b=best.r;
  const mode = b.onTop ? 'cleared'
    : (!b.up ? 'toppled'
    : (b.maxX < 0.12 ? 'never reached (trunk never past the riser)'
    : (b.feetUp<2 ? 'stalled (trunk past the riser, feet never on the tread)'
                  : 'short (feet on the tread, trunk under 95 mm above it)')));
  out[rmm] = { rise_mm:rmm, evals, seconds:+((Date.now()-t0)/1000).toFixed(0),
    bestScore:+best.score.toFixed(4), twistDir:best.dir, params:best.p,
    terminal:{onTop:b.onTop, x_mm:+(b.x*1000).toFixed(1), z_mm:+(b.z*1000).toFixed(1),
      above_mm:+(b.above*1000).toFixed(1), feetUp:b.feetUp, up:b.up},
    physics:{trunkPeakZ_mm:+(b.maxZ*1000).toFixed(1), trunkMaxX_mm:+(b.maxX*1000).toFixed(1),
      trunkMinX_mm:+(b.minX*1000).toFixed(1), trunkMaxY_mm:+(b.maxY*1000).toFixed(1),
      startX_mm:+(b.x0*1000).toFixed(1),
      headTouchedTread:b.headTouched, headContactFrac:+b.headFrac.toFixed(3),
      footOnRiserFaceFrac:+b.riserFrac.toFixed(3), outboardFootOverTread:+b.footOver.toFixed(3),
      bothFeetOverTread:+b.bothFootOver.toFixed(3), peakFeetUp:b.feetUpMax,
      maxTrunkYawWhilePlanted_rad:+b.yawPlanted.toFixed(3), uprightFrac:+b.upFrac.toFixed(3),
      ctrlSatFrac:+b.satFrac.toFixed(3),
      maxRightSoleZ_mm:+(b.maxSoleZ*1000).toFixed(1), maxRightSoleX_mm:+(b.maxSoleX*1000).toFixed(1),
      maxLeftSoleZ_mm:+(b.maxLSoleZ*1000).toFixed(1), maxLeftSoleX_mm:+(b.maxLSoleX*1000).toFixed(1)},
    failureMode:mode, offsetChecks:checks, cleared:`${cleared}/3`, improvements:log };
  console.log(`\n[${rmm}mm] DONE ${evals} evals in ${((Date.now()-t0)/1000).toFixed(0)}s best=${best.score.toFixed(3)} ${mode} cleared ${cleared}/3\n`);

  const cmd = `cd ~/projects/duck-sounds/sim && RISE=${rmm} SECONDS=${SECONDS} SEED=${process.env.SEED||777} TWIST=${TWIST} TAG=${TAG} node ../climb/search_2.mjs`;
  fs.writeFileSync(`../climb/best_2_${rmm}mm${TAG}.json`, JSON.stringify({
    move:`B2_twist_${rmm}mm`, blend:best.p.blend, approach:best.p.approach,
    gap:best.p.gap, side:best.p.side, twistDir:best.dir,
    keyframes: best.tr.map(f=>({t:+f.t.toFixed(4), pose:f.pose.map(v=>+v.toFixed(5))})),
    note:`Strategy B leg 2 (head press + trunk twist, forward-drive lever). objective `
       + `${best.score.toFixed(4)}; criterion (sim/climb_lib.mjs:150) onTop=${b.onTop}, `
       + `${cleared}/3 at start offsets -10/0/+10 mm; trunk peak z ${(b.maxZ*1000).toFixed(1)} mm, `
       + `peak x ${(b.maxX*1000).toFixed(1)} mm; head touched tread ${b.headTouched} `
       + `(contact fraction ${b.headFrac.toFixed(3)}); foot on riser face fraction `
       + `${b.riserFrac.toFixed(3)}; failure mode: ${mode}. Reproduce: ${cmd}`,
  },null,2));
  out[rmm].reproduce = cmd;
  fs.writeFileSync(`../climb/search_2-results${TAG}.json`, JSON.stringify(out,null,2));
}
console.log('WROTE search_2-results'+TAG+'.json');
