// Proof that climb/rig.mjs's run() is the same episode as sim/climb_lib.mjs's
// attempt(): same track, same opts, same rise -> byte-identical terminal state.
// cd ~/projects/duck-sounds/sim && node ../climb/parity.mjs
import { replay } from '../sim/climb_lib.mjs';
import { run, HOME } from './rig.mjs';
const a = HOME.slice(); a[5]=-1.3; a[6]=0.7; a[7]=1.4;
const track=[{t:0.5,pose:a},{t:1.6,pose:HOME.slice()}];
const opts={blend:1.6,approach:0,gap:0.03,side:0.06};
const t0=Date.now(); const A=await replay(track,opts,0.04);
const t1=Date.now(); const Bv=await run(track,opts,0.04); const t2=Date.now();
console.log('climb_lib', JSON.stringify({onTop:A.onTop,x:A.x,z:A.z,feetUp:A.feetUp,up:A.up}));
console.log('rig      ', JSON.stringify({onTop:Bv.onTop,x:Bv.x,z:Bv.z,feetUp:Bv.feetUp,up:Bv.up}));
console.log('MATCH', A.onTop===Bv.onTop && Math.abs(A.x-Bv.x)<1e-12 && Math.abs(A.z-Bv.z)<1e-12 && A.feetUp===Bv.feetUp);
console.log('ms climb_lib',t1-t0,'ms rig',t2-t1);
console.log('extras', JSON.stringify({headFrac:Bv.headFrac,upFrac:Bv.upFrac,satFrac:Bv.satFrac,footOver:Bv.footOver,yawPlanted:Bv.yawPlanted,maxZ:Bv.maxZ,x0:Bv.x0,riserFrac:Bv.riserFrac}));
