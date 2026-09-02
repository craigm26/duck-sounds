// ROUND 5 — final spot-check on the PUBLISHED bytes of climb/rig3.mjs (the
// only edit after climb/_r5_parity.mjs ran was a comment block; this proves it).
import fs from 'node:fs';
import { scoreSaved as NEW } from '../climb/rig3.mjs';
import { scoreSaved as PRE5 } from '../climb/rig3_pre_r5.mjs';
import { scoreRobust, intentHashOfFile } from '../climb/robust.mjs';
const P = '../climb/';
const R5 = new Set(['minPenetrationEpisode','minPenetrationPair','minPenetrationTick','penetrationTicksScanned','servo']);
function dd(o,n,p,out,top){ if(o===null||typeof o!=='object'){ if(!Object.is(o,n)) out.push({p,pre:o,now:n}); return; }
  if(Array.isArray(o)){ if(!Array.isArray(n)||n.length!==o.length){out.push({p:p+'.len'});return;} for(let i=0;i<o.length;i++) dd(o[i],n[i],`${p}[${i}]`,out,false); return; }
  for(const k of Object.keys(o)){ if(top&&R5.has(k))continue; if(p===''&&k==='source')continue; if(p==='.opts'&&R5.has(k))continue;
    dd(o[k], (n===null||typeof n!=='object')?undefined:n[k], `${p}.${k}`, out, false); } }
const FILES=[['best_r2_vault_40mm.json',0.040],['best_r2_vault_60mm.json',0.060],['best_r3_vault_40mm.json',0.040],
 ['best_r3_vault_50mm.json',0.050],['best_r3_vault_60mm.json',0.060],['best_r4_famA_60mm.json',0.060],
 ['ctrl_do_nothing.json',0.060],['ctrl_on_tread_60mm.json',0.060]];
let rows=0,exact=0; const bad=[];
for(const [f,h] of FILES) for(const tail of ['policy','hold']){
  const A=await NEW(P+f,{rise:h,tail}), B=await PRE5(P+f,{rise:h,tail});
  rows++; const d=[]; dd(B,A,'',d,true); if(!d.length) exact++; else bad.push({f,tail,d:d.slice(0,6)});
}
console.log(`FINAL SPOT PARITY on the published bytes: EXACT ${exact}/${rows}`);
for(const b of bad) console.log('  !! '+JSON.stringify(b));
const g=await scoreRobust(P+'best_r5_servo_60mm.json',{rise:0.060,core:true});
console.log(`best_r5_servo_60mm.json sha ${intentHashOfFile(P+'best_r5_servo_60mm.json')}`);
console.log(`   kCore=${g.kCore}/9 kCoreStable=${g.kCoreStable}/9 objectiveCore=${g.objectiveCore.toFixed(4)} meanRewardCore=${g.meanRewardCore.toFixed(4)}`);
console.log(`   servo armed in ${g.verdicts.filter(v=>v.servoArmed).length}/9 cells; servoed ticks per cell ${g.verdicts.map(v=>v.servoTicks).join(',')}`);
console.log(`   minPenetrationEpisode over the 9 cells: ${g.agg.minPenetrationEpisode_mm.toFixed(2)} mm (at-score ${g.agg.minPenetrationAtScore_mm.toFixed(2)} mm)`);
for(const v of g.verdicts) console.log(`   rise=${v.rise_mm} drop=${v.drop} f=${v.fmul} honest=${v.honest} stable=${v.stableClear} upTail=${v.uprightTailTicks}/50 penEp=${v.minPenetrationEpisode_mm}mm rew=${v.reward}`);
