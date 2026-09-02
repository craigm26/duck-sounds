// Whole-body climb: the physics estimates, before any search.
//
// Answers, from the canon plant (sim/scene.mjb) and nothing else:
//  1. friction of the foot<->step-block contact (and who wins the combine)
//  2. what vertical force ONE leg can make with the foot pressed on a vertical
//     riser face, at shin angles 30/45/60 deg -- from the real Jacobian
//     (finite-differenced through mj_forward, so it is this model's kinematics)
//  3. where the CoM is with the head pitched fully forward, and whether hooking
//     the head over an edge at rise h moves the CoM past the riser
//  4. how high/far the beak reaches, i.e. the tallest edge the head can hook
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/wholebody-physics.mjs
import load from 'mujoco';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const D = findDuckJoints(m), ADDR = findStairJoints(m);

const gname = g => m.geom(g).name || `#${g}`;
const bname = b => m.body(b).name || `#${b}`;
const jid = n => { for (let j=0;j<m.njnt;j++) if (m.jnt(j).name===n) return j; return -1; };
const bid = n => { for (let b=0;b<m.nbody;b++) if (m.body(b).name===n) return b; return -1; };
const aid = n => { for (let a=0;a<m.nu;a++) if (m.actuator(a).name===n) return a; return -1; };
const gid = n => { for (let g=0;g<m.ngeom;g++) if ((m.geom(g).name||'')===n) return g; return -1; };

// ---------------------------------------------------------------- 0. options
console.log('=== PLANT ===');
console.log(`timestep ${m.opt.timestep}s  nq=${m.nq} nv=${m.nv} nu=${m.nu} nbody=${m.nbody} ngeom=${m.ngeom}`);

// ------------------------------------------------------------- 1. mass / CoM
let mass = 0; const parts = [];
for (let b=1;b<m.nbody;b++){
  const n = bname(b);
  if (/step|ball|block|cone|wall|floor/.test(n)) continue;
  mass += m.body_mass[b]; parts.push([n, m.body_mass[b]]);
}
const W = mass*9.81;
console.log(`\nrobot mass ${(mass*1000).toFixed(1)} g   weight ${W.toFixed(3)} N`);
parts.sort((a,b)=>b[1]-a[1]);
console.log('heaviest bodies: ' + parts.slice(0,8).map(([n,v])=>`${n} ${(v*1000).toFixed(0)}g`).join(', '));

// --------------------------------------------------------------- 2. friction
console.log('\n=== FRICTION (geom_friction = [slide, spin, roll], priority, condim) ===');
const interesting = [];
for (let g=0; g<m.ngeom; g++){
  const n = gname(g);
  if (/foot_collision|sole|step|floor|wall/.test(n)) interesting.push(g);
}
for (const g of interesting.slice(0,24)){
  console.log(`  ${gname(g).padEnd(26)} fric=[${m.geom_friction[g*3].toFixed(4)}, ${m.geom_friction[g*3+1].toFixed(5)}, ${m.geom_friction[g*3+2].toFixed(6)}]  prio=${m.geom_priority[g]}  condim=${m.geom_condim[g]}  solmix=${m.geom_solmix[g].toFixed(2)}`);
}
const footG = interesting.filter(g=>/foot_collision|sole/.test(gname(g)));
const stepG = interesting.filter(g=>/step/.test(gname(g)));
if (footG.length && stepG.length){
  const f=footG[0], s=stepG[0];
  const pf=m.geom_priority[f], ps=m.geom_priority[s];
  let mu;
  if (pf>ps) mu = m.geom_friction[f*3];
  else if (ps>pf) mu = m.geom_friction[s*3];
  else mu = Math.max(m.geom_friction[f*3], m.geom_friction[s*3]);
  console.log(`\n  FOOT vs STEP: priorities ${pf} vs ${ps} -> mu_slide = ${mu.toFixed(4)}  (MuJoCo: higher priority wins outright, else elementwise max)`);
  console.log(`  head/jaw geoms:`);
  for (let g=0; g<m.ngeom; g++){
    const n=gname(g); if(!/head|jaw|beak/i.test(n)) continue;
    console.log(`    ${n.padEnd(26)} fric=[${m.geom_friction[g*3].toFixed(4)}] prio=${m.geom_priority[g]} condim=${m.geom_condim[g]} contype=${m.geom_contype[g]} conaffinity=${m.geom_conaffinity[g]}`);
  }
}

// ------------------------------------------------------- 3. torque ceilings
console.log('\n=== ACTUATOR forcerange / gear / kp ===');
for (let a=0;a<m.nu;a++){
  console.log(`  ${(m.actuator(a).name||'#'+a).padEnd(16)} range=[${m.actuator_forcerange[a*2].toFixed(4)}, ${m.actuator_forcerange[a*2+1].toFixed(4)}] gear=${m.actuator_gear[a*6].toFixed(3)} gain=${m.actuator_gainprm[a*10].toFixed(2)} bias=[${m.actuator_biasprm[a*10+1].toFixed(2)},${m.actuator_biasprm[a*10+2].toFixed(3)}]`);
}

// ------------------------------------------------- helper: set a pose, fwd
function setPose(q){                       // q: 14-vector in duck order
  mj.mj_resetData(m,d);
  if (ADDR) layoutStairs(d, ADDR, {count:0, rise:0, run:0.28, start:0.12});
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=0.12; d.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=q[i];
  mj.mj_forward(m,d);
}
const comOf = () => {                       // mass-weighted CoM of the ROBOT only
  let cx=0,cy=0,cz=0,mm=0;
  for(let b=1;b<m.nbody;b++){
    const n=bname(b); if(/step|ball|block|cone|wall|floor/.test(n)) continue;
    const w=m.body_mass[b]; mm+=w;
    cx+=w*d.xipos[b*3]; cy+=w*d.xipos[b*3+1]; cz+=w*d.xipos[b*3+2];
  }
  return [cx/mm, cy/mm, cz/mm];
};

// --------------------------------------- 4. head reach + CoM with head down
console.log('\n=== HEAD AS A HOOK ===');
const NP=5, HP=6, HY=7, HR=8;   // duck-order indices (mouth excluded)
const jaw = bid('jaw_soft'), trunk = bid('trunk_base');
const headB = bid('head')>=0?bid('head'):jaw;
setPose(HOME);
const com0 = comOf(); const trunkX0 = d.xpos[trunk*3];
console.log(`HOME:  trunk x=${(d.xpos[trunk*3]*1000).toFixed(1)} z=${(d.xpos[trunk*3+2]*1000).toFixed(1)} mm   jaw x=${(d.xpos[jaw*3]*1000).toFixed(1)} z=${(d.xpos[jaw*3+2]*1000).toFixed(1)} mm`);
console.log(`HOME:  CoM x=${(com0[0]*1000).toFixed(1)} z=${(com0[2]*1000).toFixed(1)} mm   (trunk-relative dx=${((com0[0]-trunkX0)*1000).toFixed(1)} mm)`);

const poses = {
  'neck fully DOWN (np=+1.047 lim), head lim +1.571': [NP, HI[NP], HP, HI[HP]],
  'neck fully UP   (np=-1.571 lim), head lim -1.571': [NP, LO[NP], HP, LO[HP]],
  'np=+1.047, hp=-1.571 (beak reaches FORWARD)':      [NP, HI[NP], HP, LO[HP]],
  'np=-1.571, hp=+1.571 (beak tucks under)':          [NP, LO[NP], HP, HI[HP]],
};
for (const [label, [i1,v1,i2,v2]] of Object.entries(poses)){
  const q = HOME.slice(); q[i1]=v1; q[i2]=v2;
  setPose(q);
  const c = comOf();
  console.log(`\n  ${label}`);
  console.log(`    jaw  x=${(d.xpos[jaw*3]*1000).toFixed(1)} z=${(d.xpos[jaw*3+2]*1000).toFixed(1)} mm   (dx from trunk ${((d.xpos[jaw*3]-d.xpos[trunk*3])*1000).toFixed(1)} mm, dz ${((d.xpos[jaw*3+2]-d.xpos[trunk*3+2])*1000).toFixed(1)} mm)`);
  console.log(`    CoM  x=${(c[0]*1000).toFixed(1)} z=${(c[2]*1000).toFixed(1)} mm   shift from HOME: dx=${((c[0]-com0[0])*1000).toFixed(1)} dz=${((c[2]-com0[2])*1000).toFixed(1)} mm`);
}

// Sweep neck_pitch alone: where is the beak, and how far does the CoM travel?
console.log('\n  neck_pitch sweep (head_pitch at HOME 0.349), all mm, trunk-relative:');
console.log('    np(rad)   jaw_dx   jaw_dz   CoM_dx   CoM_dz');
for (let np=LO[NP]; np<=HI[NP]+1e-9; np+=(HI[NP]-LO[NP])/10){
  const q=HOME.slice(); q[NP]=np; setPose(q); const c=comOf();
  console.log(`    ${np.toFixed(3).padStart(6)}  ${((d.xpos[jaw*3]-d.xpos[trunk*3])*1000).toFixed(1).padStart(7)} ${((d.xpos[jaw*3+2]-d.xpos[trunk*3+2])*1000).toFixed(1).padStart(8)} ${((c[0]-com0[0])*1000).toFixed(1).padStart(8)} ${((c[2]-com0[2])*1000).toFixed(1).padStart(8)}`);
}

// -------------------------------- 5. neck lever: force available at the beak
const neckJ = jid('neck_pitch'), headJ = jid('head_pitch');
setPose(HOME);
const anchorOf = j => [d.xanchor[j*3], d.xanchor[j*3+1], d.xanchor[j*3+2]];
const armTo = (p,b) => Math.hypot(d.xpos[b*3]-p[0], d.xpos[b*3+1]-p[1], d.xpos[b*3+2]-p[2]);
const neckTq = m.actuator_forcerange[aid('neck_pitch')*2+1];
const headTq = m.actuator_forcerange[aid('head_pitch')*2+1];
console.log(`\n=== NECK LEVER (HOME) ===`);
console.log(`  neck_pitch anchor -> jaw ${(armTo(anchorOf(neckJ),jaw)*1000).toFixed(1)} mm, tau ${neckTq.toFixed(4)} N.m -> F_max ${(neckTq/armTo(anchorOf(neckJ),jaw)).toFixed(3)} N = ${(neckTq/armTo(anchorOf(neckJ),jaw)/W*100).toFixed(0)}% of weight`);
console.log(`  head_pitch anchor -> jaw ${(armTo(anchorOf(headJ),jaw)*1000).toFixed(1)} mm, tau ${headTq.toFixed(4)} N.m -> F_max ${(headTq/armTo(anchorOf(headJ),jaw)).toFixed(3)} N = ${(headTq/armTo(anchorOf(headJ),jaw)/W*100).toFixed(0)}% of weight`);
// and at the head-down pose, where the lever is what matters
{ const q=HOME.slice(); q[NP]=HI[NP]; q[HP]=LO[HP]; setPose(q);
  const an=anchorOf(neckJ), ah=anchorOf(headJ);
  console.log(`  head-down pose: neck arm ${(armTo(an,jaw)*1000).toFixed(1)} mm -> ${(neckTq/armTo(an,jaw)).toFixed(3)} N (${(neckTq/armTo(an,jaw)/W*100).toFixed(0)}%);  head arm ${(armTo(ah,jaw)*1000).toFixed(1)} mm -> ${(headTq/armTo(ah,jaw)).toFixed(3)} N (${(headTq/armTo(ah,jaw)/W*100).toFixed(0)}%)`);
  // HORIZONTAL arm is what converts neck torque into a VERTICAL force at the beak
  console.log(`  horizontal lever neck_anchor->jaw = ${(Math.abs(d.xpos[jaw*3]-an[0])*1000).toFixed(1)} mm -> vertical F at beak = ${(neckTq/Math.max(Math.abs(d.xpos[jaw*3]-an[0]),1e-6)).toFixed(3)} N (${(neckTq/Math.max(Math.abs(d.xpos[jaw*3]-an[0]),1e-6)/W*100).toFixed(0)}% of weight)`);
}

// ---------------------------- 6. one leg pushing on a VERTICAL riser face
// Finite-difference Jacobian of the LEFT foot geom wrt (hip_pitch, knee, ankle),
// then: what force set {fx, fz} at the foot is reachable with |tau| <= 0.6405?
console.log('\n=== ONE LEG ON A VERTICAL FACE ===');
const LHP=2, LK=3, LA=4;
const lfoot = footG.find(g=>/left/.test(gname(g))) ?? footG[0];
console.log(`  foot geom used: ${gname(lfoot)}`);
const TAU = 0.6405;
function footPos(q){ setPose(q); return [d.geom_xpos[lfoot*3], d.geom_xpos[lfoot*3+1], d.geom_xpos[lfoot*3+2]]; }
function jac3(q){                              // d(foot x,z) / d(hip,knee,ankle)
  const h=1e-5, p0=footPos(q), J=[];
  for (const k of [LHP,LK,LA]){
    const qq=q.slice(); qq[k]+=h; const p1=footPos(qq);
    J.push([(p1[0]-p0[0])/h, (p1[2]-p0[2])/h]);
  }
  return {J, p0};
}
// shin angle from vertical: build a pose by choosing hip/knee so the shin
// (knee->ankle segment) makes the wanted angle, foot forward of the hip.
const kneeB = bid('left_knee')>=0?bid('left_knee'):-1;
function shinAngle(q){                          // deg from VERTICAL, sagittal
  setPose(q);
  const kj = jid('left_knee'), aj = jid('left_ankle');
  const k = anchorOf(kj), a = anchorOf(aj);
  const dx=a[0]-k[0], dz=a[2]-k[2];
  return Math.atan2(Math.abs(dx), Math.abs(dz))*180/Math.PI;
}
console.log('  shin_deg  hip_pitch  knee   foot(x,z)mm   Fz_max_up(N)  Fx_max_push(N)  Fz@Fx=Fz/mu(N)');
for (const target of [30,45,60]){
  // crude 1-D search over knee with hip chosen to keep the foot at the wall
  let best=null;
  for (let hip=-1.4; hip<=1.4; hip+=0.05){
    for (let knee=-1.5; knee<=1.5; knee+=0.05){
      const q=HOME.slice(); q[LHP]=hip; q[LK]=knee; q[LA]=HOME[LA];
      const s=shinAngle(q);
      const e=Math.abs(s-target);
      if(!best||e<best.e) best={e,s,hip,knee,q};
    }
  }
  const {J,p0}=jac3(best.q);
  // tau = J^T f  (J rows = per-dof gradient [dx,dz]); tau_i = J[i]·f
  // max fz s.t. |J[i]·f| <= TAU for all i, with fx free (a wall gives normal fx)
  // scan force directions
  let fzUp=0, fxPush=0;
  const feas = f => J.every(r=>Math.abs(r[0]*f[0]+r[1]*f[1])<=TAU+1e-12);
  const scale = f => { let lo=0, hi=500; for(let it=0;it<60;it++){const mid=(lo+hi)/2; if(feas([f[0]*mid,f[1]*mid])) lo=mid; else hi=mid;} return lo; };
  // pure vertical
  fzUp = scale([0,1]);
  fxPush = scale([1,0]);
  // and the combination a wall actually needs: fx = fz/mu (friction-limited)
  const mu = m.geom_friction[lfoot*3];
  const dirn = [1/mu, 1]; const nrm=Math.hypot(dirn[0],dirn[1]);
  const s2 = scale([dirn[0]/nrm, dirn[1]/nrm]);
  console.log(`  ${best.s.toFixed(1).padStart(7)}  ${best.hip.toFixed(2).padStart(9)} ${best.knee.toFixed(2).padStart(6)}  ${(p0[0]*1000).toFixed(0)},${(p0[2]*1000).toFixed(0)}`.padEnd(58)
    + `${fzUp.toFixed(2).padStart(10)}  ${fxPush.toFixed(2).padStart(13)}  ${(s2*dirn[1]/nrm).toFixed(2).padStart(12)}`);
}
console.log(`  (Fz_max_up = the vertical force one leg's three sagittal joints can hold at the foot,`);
console.log(`   from tau=J^T f with |tau| <= ${TAU} N.m on EVERY joint. Robot weight ${W.toFixed(2)} N.)`);

// --------------------------------- 7. how high can a foot reach from a stance?
console.log('\n=== FOOT REACH (the ceiling that made step_up cap out) ===');
setPose(HOME);
const zHomeFoot = d.geom_xpos[lfoot*3+2];
let hi=-1, hiq=null;
for (let hip=-1.57; hip<=1.57; hip+=0.03)
  for (let knee=-1.57; knee<=1.57; knee+=0.03){
    const q=HOME.slice(); q[LHP]=hip; q[LK]=knee;
    setPose(q);
    const zz=d.geom_xpos[lfoot*3+2], xx=d.geom_xpos[lfoot*3];
    if (xx>d.xpos[trunk*3] && zz>hi){ hi=zz; hiq={hip,knee,xx,zz}; }
  }
console.log(`  home foot geom z = ${(zHomeFoot*1000).toFixed(1)} mm (trunk free at z=120mm, no gravity settle)`);
console.log(`  highest the LEFT foot geom gets FORWARD of the trunk: z=${(hi*1000).toFixed(1)} mm at hip=${hiq.hip.toFixed(2)} knee=${hiq.knee.toFixed(2)}, x=${(hiq.xx*1000).toFixed(1)} mm`);
console.log(`  i.e. relative to the standing foot, a swing foot can gain ${((hi-zHomeFoot)*1000).toFixed(1)} mm of height in this pose family.`);

// --------------------------------- 8. hip yaw/roll: how far can a hip TWIST?
console.log('\n=== TRUNK TWIST BUDGET (hip yaw / roll) ===');
for (const nm of ['left_hip_yaw','left_hip_roll','right_hip_yaw','right_hip_roll','head_yaw','head_roll']){
  const j=jid(nm);
  console.log(`  ${nm.padEnd(16)} range [${m.jnt_range[j*2].toFixed(4)}, ${m.jnt_range[j*2+1].toFixed(4)}] rad = [${(m.jnt_range[j*2]*180/Math.PI).toFixed(1)}, ${(m.jnt_range[j*2+1]*180/Math.PI).toFixed(1)}] deg`);
}
// what does yawing both hips to their limits do to the foot positions?
{
  const q=HOME.slice(); q[0]=HI[0]; q[10]=LO[10];    // left yaw +, right yaw -
  setPose(q);
  const rf = footG.find(g=>/right/.test(gname(g)));
  console.log(`  both hips yawed to limit: left foot (${(d.geom_xpos[lfoot*3]*1000).toFixed(1)}, ${(d.geom_xpos[lfoot*3+1]*1000).toFixed(1)}) mm, right foot (${(d.geom_xpos[rf*3]*1000).toFixed(1)}, ${(d.geom_xpos[rf*3+1]*1000).toFixed(1)}) mm`);
  const q2=HOME.slice(); q2[1]=HI[1]; q2[11]=HI[11];  // both rolls +
  setPose(q2);
  console.log(`  both hip rolls at +limit: left foot y=${(d.geom_xpos[lfoot*3+1]*1000).toFixed(1)} mm, right foot y=${(d.geom_xpos[rf*3+1]*1000).toFixed(1)} mm  (HOME y left ${(()=>{setPose(HOME);return (d.geom_xpos[lfoot*3+1]*1000).toFixed(1);})()})`);
}
