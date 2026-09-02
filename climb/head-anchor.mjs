// The head as a hook: what it can actually touch, and what it can actually hold.
//
// geom-census.mjs corrected a false negative: the jaw's three collision meshes
// are UNNAMED, so a name regex found nothing. They exist, contype/conaffinity
// 5/5, and 5 & 4 (a step block) is non-zero -- the head DOES collide with a
// tread. Everything below follows from that.
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
const bid = n => { for (let b=0;b<m.nbody;b++) if (m.body(b).name===n) return b; return -1; };
const TAU = 0.6405, MASS = (()=>{let s=0;for(let b=1;b<m.nbody;b++){const n=m.body(b).name||'';if(/step|ball|block|cone|wall|floor/.test(n))continue;s+=m.body_mass[b];}return s;})();
const W = MASS*9.81;

console.log('=== JAW COLLISION GEOMS (the hook) ===');
const jawB = bid('jaw_soft'); const JAWG=[];
for(let g=0;g<m.ngeom;g++) if(m.geom_bodyid[g]===jawB){ JAWG.push(g);
  console.log(`  geom ${g}: fric=[${m.geom_friction[g*3].toFixed(4)}, ${m.geom_friction[g*3+1].toFixed(5)}, ${m.geom_friction[g*3+2].toFixed(6)}] prio=${m.geom_priority[g]} condim=${m.geom_condim[g]} contype/conaff=${m.geom_contype[g]}/${m.geom_conaffinity[g]} rbound=${(m.geom_rbound[g]*1000).toFixed(1)}mm`);
}
const stepG = (()=>{for(let g=0;g<m.ngeom;g++) if((m.geom(g).name||'')==='step0_geom') return g;})();
console.log(`  step0_geom fric ${m.geom_friction[stepG*3].toFixed(3)} prio ${m.geom_priority[stepG]}`);
{
  const jg=JAWG[0], pf=m.geom_priority[jg], ps=m.geom_priority[stepG];
  const mu = pf>ps ? m.geom_friction[jg*3] : ps>pf ? m.geom_friction[stepG*3] : Math.max(m.geom_friction[jg*3], m.geom_friction[stepG*3]);
  console.log(`  HEAD vs STEP mu_slide = ${mu.toFixed(3)}   robot weight ${W.toFixed(3)} N`);
  console.log(`  -> to hold ${W.toFixed(2)} N by head friction alone needs normal N >= ${(W/mu).toFixed(2)} N pressed into the face.`);
}

// ---------------------------------------------------------------- settle
function settleFlat(){                      // free fall onto the floor, HOME pose held
  mj.mj_resetData(m,d);
  layoutStairs(d, ADDR, {count:0, rise:0, run:0.28, start:0.12});
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=0.12; d.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){d.qpos[D.qpos[i]]=HOME[i]; d.ctrl[i]=HOME[i];}
  mj.mj_forward(m,d);
  for(let t=0;t<600;t++){ layoutStairs(d, ADDR, {count:0,rise:0,run:0.28,start:0.12}); mj.mj_step(m,d); }
}
settleFlat();
const trunk=bid('trunk_base');
const zTrunkStand = d.qpos[D.freeQpos+2];
console.log(`\n=== SETTLED, HOME POSE, FLAT FLOOR (ctrl=HOME, no policy) ===`);
console.log(`  trunk free-joint z = ${(zTrunkStand*1000).toFixed(1)} mm   trunk body z = ${(d.xpos[trunk*3+2]*1000).toFixed(1)} mm`);
let comz=0,comx=0,mm=0; for(let b=1;b<m.nbody;b++){const n=m.body(b).name||'';if(/step|ball|block|cone|wall|floor/.test(n))continue;mm+=m.body_mass[b];comx+=m.body_mass[b]*d.xipos[b*3];comz+=m.body_mass[b]*d.xipos[b*3+2];}
console.log(`  CoM z = ${(comz/mm*1000).toFixed(1)} mm, CoM x = ${(comx/mm*1000).toFixed(1)} mm  (feet x ~ ${(d.geom_xpos[29*3]*1000).toFixed(1)} mm)`);
console.log(`  jaw_soft z = ${(d.xpos[jawB*3+2]*1000).toFixed(1)} mm, x = ${(d.xpos[jawB*3]*1000).toFixed(1)} mm`);

// ------------------------------------------- JAW WORKSPACE, trunk pinned
function setPose(q, zTrunk){
  mj.mj_resetData(m,d);
  layoutStairs(d, ADDR, {count:0,rise:0,run:0.28,start:0.12});
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=zTrunk; d.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=q[i];
  mj.mj_forward(m,d);
}
const NP=5,HP=6,HY=7,HR=8;
console.log(`\n=== JAW WORKSPACE (trunk pinned upright at the settled standing z=${(zTrunkStand*1000).toFixed(0)} mm) ===`);
let far=null, high=null, lowFar=null;
const pts=[];
for(let np=LO[NP]; np<=HI[NP]+1e-9; np+=0.05)
  for(let hp=LO[HP]; hp<=HI[HP]+1e-9; hp+=0.05){
    const q=HOME.slice(); q[NP]=np; q[HP]=hp; setPose(q, zTrunkStand);
    const x=d.xpos[jawB*3], z=d.xpos[jawB*3+2];
    pts.push([x,z,np,hp]);
    if(!far||x>far[0]) far=[x,z,np,hp];
    if(!high||z>high[1]) high=[x,z,np,hp];
  }
console.log(`  jaw reaches FURTHEST forward at x=${(far[0]*1000).toFixed(1)} mm, z=${(far[1]*1000).toFixed(1)} mm  (neck_pitch ${far[2].toFixed(2)}, head_pitch ${far[3].toFixed(2)})`);
console.log(`  jaw reaches HIGHEST at z=${(high[1]*1000).toFixed(1)} mm, x=${(high[0]*1000).toFixed(1)} mm  (neck_pitch ${high[2].toFixed(2)}, head_pitch ${high[3].toFixed(2)})`);
console.log('  jaw centre positions on the forward frontier (x >= 60 mm), by height band:');
for(const [lo,hi2] of [[0,0.02],[0.02,0.04],[0.04,0.06],[0.06,0.09],[0.09,0.12],[0.12,0.18],[0.18,0.30]]){
  const band = pts.filter(p=>p[1]>=lo&&p[1]<hi2);
  if(!band.length) continue;
  const b = band.reduce((a,p)=>p[0]>a[0]?p:a);
  console.log(`    z in [${(lo*1000).toFixed(0)},${(hi2*1000).toFixed(0)}) mm : max jaw x = ${(b[0]*1000).toFixed(1)} mm at z=${(b[1]*1000).toFixed(1)} (np ${b[2].toFixed(2)}, hp ${b[3].toFixed(2)})`);
}
console.log(`  NOTE jaw geom rbound is ${(m.geom_rbound[JAWG[0]]*1000).toFixed(0)}/${(m.geom_rbound[JAWG[1]]*1000).toFixed(0)}/${(m.geom_rbound[JAWG[2]]*1000).toFixed(0)} mm, so the CONTACT surface reaches further than the body origin above.`);
// the actual forward-most jaw GEOM surface, using geom_xpos +/- rbound
{
  let bestg=null;
  for(let np=LO[NP]; np<=HI[NP]+1e-9; np+=0.05) for(let hp=LO[HP]; hp<=HI[HP]+1e-9; hp+=0.05){
    const q=HOME.slice(); q[NP]=np; q[HP]=hp; setPose(q,zTrunkStand);
    for(const g of JAWG){ const x=d.geom_xpos[g*3]+m.geom_rbound[g], z=d.geom_xpos[g*3+2];
      if(!bestg||x>bestg[0]) bestg=[x,z,np,hp,g]; }
  }
  console.log(`  furthest jaw-geom SURFACE (xpos+rbound): x=${(bestg[0]*1000).toFixed(1)} mm at z=${(bestg[1]*1000).toFixed(1)} mm (np ${bestg[2].toFixed(2)}, hp ${bestg[3].toFixed(2)}, geom ${bestg[4]})`);
}

// -------------------- FORCE AVAILABLE AT THE JAW, from the 4 neck/head joints
console.log('\n=== FORCE THE NECK CAN HOLD AT THE JAW (tau = J^T f, |tau| <= 0.6405 on all 4) ===');
function jawJac(q){
  const h=1e-5; setPose(q,zTrunkStand);
  const p0=[d.xpos[jawB*3], d.xpos[jawB*3+1], d.xpos[jawB*3+2]];
  const rows=[];
  for(const k of [NP,HP,HY,HR]){
    const qq=q.slice(); qq[k]+=h; setPose(qq,zTrunkStand);
    rows.push([(d.xpos[jawB*3]-p0[0])/h, (d.xpos[jawB*3+1]-p0[1])/h, (d.xpos[jawB*3+2]-p0[2])/h]);
  }
  return {rows,p0};
}
const feas = (rows,f) => rows.every(r=>Math.abs(r[0]*f[0]+r[1]*f[1]+r[2]*f[2])<=TAU+1e-12);
const scale = (rows,dir) => { let lo=0,hi=2000; for(let i=0;i<60;i++){const mid=(lo+hi)/2; if(feas(rows,dir.map(v=>v*mid))) lo=mid; else hi=mid;} return lo; };
console.log('  np     hp     jaw(x,z)mm      Fz_up(N)  %W    Fx_back(N)  Fz_down(N)');
for (const [np,hp] of [[LO[NP],LO[HP]],[LO[NP],0],[LO[NP],HI[HP]],[-1.0,-0.8],[-0.6,-0.4],[0,HOME[HP]],[HI[NP],HI[HP]]]){
  const q=HOME.slice(); q[NP]=np; q[HP]=hp;
  const {rows,p0}=jawJac(q);
  const up=scale(rows,[0,0,1]), back=scale(rows,[-1,0,0]), down=scale(rows,[0,0,-1]);
  console.log(`  ${np.toFixed(2).padStart(5)} ${hp.toFixed(2).padStart(6)}  ${(p0[0]*1000).toFixed(0).padStart(5)},${(p0[2]*1000).toFixed(0).padStart(5)}   `
    +`${up.toFixed(2).padStart(9)}  ${(up/W*100).toFixed(0).padStart(4)}  ${back.toFixed(2).padStart(9)}  ${down.toFixed(2).padStart(10)}`);
}
console.log(`  (Fz_up = the upward force the four neck/head joints can react at the jaw, i.e. how much`);
console.log(`   of the ${W.toFixed(2)} N body the head can hold when the jaw is pressed onto a tread.)`);

// ------------------- CoM vs the riser plane: the whole game, per rise
console.log('\n=== CoM PAST THE RISER? (trunk pitched forward about the ankles, head hooked) ===');
console.log('  Pitching the whole trunk is a FREE-JOINT rotation, not a servo: the duck gets it by');
console.log('  driving both hip pitches. Below: CoM x for a trunk pitched theta forward about the');
console.log('  ankle line, with the neck at its most-forward setting. The riser is at x = start = 120 mm');
console.log('  and the duck starts with its feet ~ (120 - 70 - gap) mm behind it.');
console.log('  theta_deg  CoM_x(mm)  CoM_z(mm)  jaw_x(mm)  jaw_z(mm)');
for(const th of [0,10,20,30,45,60,75,90]){
  const q=HOME.slice(); q[NP]=LO[NP]; q[HP]=-0.6;
  mj.mj_resetData(m,d);
  layoutStairs(d, ADDR, {count:0,rise:0,run:0.28,start:0.12});
  const a=-th*Math.PI/180;   // pitch forward about +y axis
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=zTrunkStand;
  d.qpos[D.freeQpos+3]=Math.cos(a/2); d.qpos[D.freeQpos+4]=0; d.qpos[D.freeQpos+5]=Math.sin(a/2); d.qpos[D.freeQpos+6]=0;
  for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=q[i];
  mj.mj_forward(m,d);
  let cx=0,cz=0,mmm=0; for(let b=1;b<m.nbody;b++){const n=m.body(b).name||'';if(/step|ball|block|cone|wall|floor/.test(n))continue;mmm+=m.body_mass[b];cx+=m.body_mass[b]*d.xipos[b*3];cz+=m.body_mass[b]*d.xipos[b*3+2];}
  console.log(`  ${String(th).padStart(9)}  ${(cx/mmm*1000).toFixed(1).padStart(9)}  ${(cz/mmm*1000).toFixed(1).padStart(9)}  ${(d.xpos[jawB*3]*1000).toFixed(1).padStart(9)}  ${(d.xpos[jawB*3+2]*1000).toFixed(1).padStart(9)}`);
}
