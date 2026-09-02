// Strategy B — HEAD PRESS + TRUNK TWIST. Random-restart hill climbing over a
// 42-parameter, 6-keyframe skeleton plus a searched twist direction.
//
// Run from sim/ (climb_lib/rig read scene.mjb, duckkit-constants.json and the
// ONNX by cwd-relative path):
//     cd ~/projects/duck-sounds/sim && RISE=40 SECONDS=300 node ../climb/search_1.mjs
//
// Env: RISE (mm, comma list ok), SECONDS (wall budget per rise), SEED, TWIST
// (+1 / -1 / both), TAG (output suffix).
import fs from 'node:fs';
import { run, HOME, LO, HI } from './rig.mjs';

const J = { lhy:0, lhr:1, lhp:2, lk:3, la:4, np:5, hp:6, hy:7, hr:8,
            rhy:9, rhr:10, rhp:11, rk:12, ra:13 };

// ── the searched box ────────────────────────────────────────────────────────
// Where the strategy's own sign convention disagreed with the harness's
// (ladder.mjs trackOf uses +offset on the LEFT and -offset on the RIGHT for the
// same physical motion), the bound is WIDENED rather than guessed.
const B = {
  t1:[0.25,0.55], d2:[0.15,0.40], d3:[0.15,0.45], d4:[0.15,0.45], d5:[0.15,0.50], d6:[0.25,0.70],
  // F1 PRELOAD / PREWIND (legs antisymmetric: left +v, right -v)
  hp1:[0.70,1.40], k1:[-1.20,0.60], a1:[-1.20,0.90], hy1:[0.10,0.4363],
  neck1:[0.00,0.35], head1:[0.20,0.60],
  // F2 PRESS
  neck2:[-1.5708,-1.20], head2:[0.40,1.10], roll2:[-0.4363,0.4363],
  // F3 WIND
  yaw3:[0.60,2.20], hy3:[0.0,0.4363], hr3:[-0.30,0.38],
  neck3:[-1.5708,-1.10], head3:[0.40,1.20],
  // F4 STEM — inboard = LEFT (the wall is at +y, i.e. the duck's left facing +x)
  inHip4:[1.10,1.5708], inKnee4:[-1.30,-0.55], inAnk4:[0.85,1.45], inRoll4:[-0.384,0.384],
  // F5 OUTBOARD (RIGHT) FOOT ONTO THE TREAD
  outHip5:[-1.5708,1.5708], outKnee5:[-1.5708,1.5708], outAnk5:[-1.00,1.00], outRoll5:[-0.384,0.384],
  yaw5:[0.30,2.20], neck5:[-1.5708,-0.80], head5:[0.20,1.20],
  // F6 UNWIND + HIP-OVER
  yaw6:[-0.40,0.40], neck6:[0.40,1.0472], head6:[-1.5708,-0.60], roll6:[-0.4363,0.4363],
  hy6:[-0.20,0.20], hp6:[-1.40,1.40], k6:[-1.40,1.40], a6:[-1.00,1.00],
  // opts
  blend:[0.8,2.4], gap:[0.020,0.045], side:[0.0,0.085],
};
const KEYS = Object.keys(B);

function clampJ(pose){ return pose.map((v,k)=>Math.min(Math.max(v,LO[k]),HI[k])); }

/** 6 searched frames + a HOME tail. `dir` is twistDir. */
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
  F3[J.hy]  = dir*p.yaw3;
  F3[J.lhy] = -dir*p.hy3;  F3[J.rhy] = dir*p.hy3;
  F3[J.lhr] = p.hr3;       F3[J.rhr] = -p.hr3;
  F3[J.np]  = p.neck3;     F3[J.hp]  = p.head3;

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
function reward(r, h){
  const R =
      3.0*cl((r.maxX - r.x0)/0.20,0,1)
    + 3.0*cl((r.maxZ - h)/0.12,0,1)
    + 2.0*r.headFrac
    + 2.0*r.footOver
    + 1.0*cl(r.yawPlanted/0.9,0,1)*r.headFrac
    + 1.0*r.upFrac
    - 5.0*Math.max(0, r.x0 - r.minX)/0.10
    - 2.0*Math.max(0, r.maxY - 1.44)/0.01
    - 0.5*r.satFrac
    + (r.onTop?100:0);
  return R;
}

// ── deterministic RNG so a run is reproducible from its SEED ────────────────
let S = (+process.env.SEED || 12345)>>>0;
const rnd = () => { S ^= S<<13; S>>>=0; S ^= S>>>17; S ^= S<<5; S>>>=0; return S/4294967296; };
const gauss = () => { let u=0,v=0; while(!u)u=rnd(); while(!v)v=rnd();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

function randP(dir){
  const p={}; for(const k of KEYS){ const [a,b]=B[k]; p[k]=a+rnd()*(b-a); }
  p.side = dir>0 ? 0.045+rnd()*0.040 : rnd()*0.030;      // coupled to twistDir
  return p;
}
function jitter(p, s, dir){
  const q={}; for(const k of KEYS){ const [a,b]=B[k]; q[k]=cl(p[k]+gauss()*(b-a)*s,a,b); }
  q.side = dir>0 ? cl(q.side,0.045,0.085) : cl(q.side,0.0,0.030);
  return q;
}

const RISES = (process.env.RISE||'40').split(',').map(s=>+s);
const SECONDS = +(process.env.SECONDS||300);
const TWIST = process.env.TWIST||'both';
const TAG = process.env.TAG||'';
const DIRS = TWIST==='both'?[+1,-1]:[+TWIST];

const evalP = async (p,dir,h) => {
  const tr = trackOf(p,dir);
  const r = await run(tr, {blend:p.blend, gap:p.gap, side:p.side, approach:0}, h);
  return { r, score: reward(r,h), tr };
};

const out = {};
for (const rmm of RISES){
  const h = rmm/1000;
  const t0 = Date.now();
  let best = null, evals = 0;
  const log = [];
  while ((Date.now()-t0)/1000 < SECONDS){
    const dir = DIRS[evals % DIRS.length];
    let p = randP(dir);
    let cur = await evalP(p,dir,h); evals++;
    if (!best || cur.score>best.score){ best={...cur,p,dir};
      log.push({evals,score:+cur.score.toFixed(3),dir,r:cur.r});
      console.log(`[${rmm}mm] e${evals} NEW ${cur.score.toFixed(3)} dir${dir>0?'+':'-'} onTop=${cur.r.onTop} z=${(cur.r.z*1000).toFixed(0)} peakZ=${(cur.r.maxZ*1000).toFixed(0)} head=${cur.r.headFrac.toFixed(2)} foot=${cur.r.footOver.toFixed(2)} yaw=${cur.r.yawPlanted.toFixed(2)} feetUp=${cur.r.feetUp}`);
    }
    // hill climb from this restart
    let sigma=0.16, stall=0;
    while (stall<14 && (Date.now()-t0)/1000 < SECONDS){
      const q = jitter(p, sigma, dir);
      const c = await evalP(q,dir,h); evals++;
      if (c.score > cur.score + 1e-6){ cur=c; p=q; stall=0; sigma=Math.min(0.22,sigma*1.15);
        if (cur.score>best.score){ best={...cur,p:{...p},dir};
          log.push({evals,score:+cur.score.toFixed(3),dir,r:cur.r});
          console.log(`[${rmm}mm] e${evals} NEW ${cur.score.toFixed(3)} dir${dir>0?'+':'-'} onTop=${cur.r.onTop} z=${(cur.r.z*1000).toFixed(0)} peakZ=${(cur.r.maxZ*1000).toFixed(0)} head=${cur.r.headFrac.toFixed(2)} foot=${cur.r.footOver.toFixed(2)} yaw=${cur.r.yawPlanted.toFixed(2)} feetUp=${cur.r.feetUp} riser=${cur.r.riserFrac.toFixed(2)}`);
        }
      } else { stall++; sigma=Math.max(0.03,sigma*0.90); }
    }
  }
  // determinism / robustness re-run at start offsets -10 / 0 / +10 mm on gap
  const offs=[-0.010,0,0.010]; const checks=[];
  for (const d of offs){
    const r = await run(best.tr, {blend:best.p.blend, gap:Math.max(0.01,best.p.gap+d), side:best.p.side, approach:0}, h);
    checks.push({off:d, onTop:r.onTop, z:+(r.z*1000).toFixed(1), above:+(r.above*1000).toFixed(1),
                 feetUp:r.feetUp, up:r.up, peakZ:+(r.maxZ*1000).toFixed(1), maxX:+(r.maxX*1000).toFixed(1),
                 headFrac:+r.headFrac.toFixed(3), riserFrac:+r.riserFrac.toFixed(3), footOver:+r.footOver.toFixed(3)});
  }
  const cleared = checks.filter(c=>c.onTop).length;
  const b=best.r;
  const mode = b.onTop ? 'cleared'
    : (!b.up ? 'toppled'
    : (b.maxX < 0.12 ? 'never reached (trunk never past the riser)'
    : (b.feetUp<2 ? 'stalled (trunk past the riser, feet never on the tread)' : 'short (feet up but trunk too low)')));
  out[rmm] = { rise_mm:rmm, evals, seconds:+((Date.now()-t0)/1000).toFixed(0),
    bestScore:+best.score.toFixed(4), twistDir:best.dir, params:best.p,
    terminal:{onTop:b.onTop, x_mm:+(b.x*1000).toFixed(1), z_mm:+(b.z*1000).toFixed(1),
      above_mm:+(b.above*1000).toFixed(1), feetUp:b.feetUp, up:b.up},
    physics:{trunkPeakZ_mm:+(b.maxZ*1000).toFixed(1), trunkMaxX_mm:+(b.maxX*1000).toFixed(1),
      trunkMinX_mm:+(b.minX*1000).toFixed(1), trunkMaxY_mm:+(b.maxY*1000).toFixed(1),
      headTouchedTread:b.headTouched, headContactFrac:+b.headFrac.toFixed(3),
      footOnRiserFaceFrac:+b.riserFrac.toFixed(3), outboardFootOverTread:+b.footOver.toFixed(3),
      maxTrunkYawWhilePlanted_rad:+b.yawPlanted.toFixed(3), uprightFrac:+b.upFrac.toFixed(3),
      ctrlSatFrac:+b.satFrac.toFixed(3), maxRightSoleZ_mm:+(b.maxSoleZ*1000).toFixed(1),
      maxRightSoleX_mm:+(b.maxSoleX*1000).toFixed(1)},
    failureMode:mode, offsetChecks:checks, cleared:`${cleared}/3`, improvements:log.length };
  console.log(`\n[${rmm}mm] DONE ${evals} evals in ${((Date.now()-t0)/1000).toFixed(0)}s  best=${best.score.toFixed(3)}  ${mode}  cleared ${cleared}/3\n`);
  fs.writeFileSync(`../climb/best_1_${rmm}mm${TAG}.json`, JSON.stringify({
    move:`B_twist_${rmm}mm`, blend:best.p.blend, approach:0, gap:best.p.gap, side:best.p.side,
    keyframes: best.tr.map(f=>({t:+f.t.toFixed(4), pose:f.pose.map(v=>+v.toFixed(5))})),
    note:`Strategy B (head press + trunk twist), twistDir=${best.dir}. objective ${best.score.toFixed(4)}, `
       + `onTop=${b.onTop}, criterion ${cleared}/3 at gap offsets -10/0/+10 mm. `
       + `Reproduce: cd ~/projects/duck-sounds/sim && RISE=${rmm} SECONDS=${SECONDS} SEED=${process.env.SEED||12345} TWIST=${TWIST} node ../climb/search_1.mjs`,
  },null,2));
  fs.writeFileSync(`../climb/search_1-results${TAG}.json`, JSON.stringify(out,null,2));
}
console.log(JSON.stringify(out,null,2).slice(0,200));
