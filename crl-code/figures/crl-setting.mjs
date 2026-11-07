/** Original deterministic maze calculations; no dependencies or external data. */
export const modeAt=cycle=>Math.floor(((cycle-1)%24)/8)===1?'B':'A';
export const goodSide=mode=>mode==='A'?'L':'R';
export function runMaze({cycles=24,startSide='L',update=true,mode=null,oracle=false}={}){
 let side=startSide,total=0;const circles=[],steps=[{step:0,cumulative:0}],snapshots=[{cycle:0,side}];
 for(let k=1;k<=cycles;k++){
  const currentMode=mode??modeAt(k),chosen=oracle?goodSide(currentMode):side;
  const reward=Number(chosen===goodSide(currentMode)),before=side;
  total+=reward;steps.push({step:2*k-1,reward,cumulative:total,position:chosen});
  if(update&&!oracle&&reward===0)side=side==='L'?'R':'L';
  circles.push({cycle:k,mode:currentMode,action:chosen,reward,sideBefore:before,sideAfter:side});
  steps.push({step:2*k,reward:0,cumulative:total,position:'J'});snapshots.push({cycle:k,side});
 }
 return {circles,steps,snapshots,total,perPrimitiveStep:total/(2*cycles),finalSide:side};
}
export function computeCRLSetting(){
 const adaptive=runMaze(),frozen=runMaze({update:false}),oracle=runMaze({oracle:true});
 const at8=adaptive.snapshots[8].side,at16=adaptive.snapshots[16].side;
 const continuedB=runMaze({cycles:8,startSide:at8,mode:'B'}),frozenB=runMaze({cycles:8,startSide:at8,mode:'B',update:false});
 const freshB=runMaze({cycles:8,startSide:'L',mode:'B'});
 const probeA=side=>Number(side==='L');
 return {schema_version:1,provenance:{kind:'deterministic original teaching calculation',generator:'scripts/generate-crl-setting.mjs',copiedSourceImages:false,seed:null,seedReason:'no random numbers; a fixed schedule and deterministic learner'},world:{cells:{L:[0,1],J:[1,1],R:[2,1]},start:'J',reward:'J to chosen side: 1 iff side matches mode; forced return to J: 0',actions:{J:['L','R'],L:['return'],R:['return']},schedule:'A eight circles, B eight, A eight; repeat the 24-circle period',budget:{circles:24,primitiveSteps:48},visible:'position and obtained reward; no mode or switch marker',externalReset:false,fullState:'position plus hidden circle index modulo 24; index advances on return to J'},learner:{name:'win-stay / lose-switch',initialSide:'L',state:'one persistent side bit; at a side cell reward 0 flips it, reward 1 retains it',data:'only the reward of the chosen side; no replay, oracle query, gradient, optimizer or learned model',knownStructure:'binary deterministic rewards and exactly one good side',limit:'noise or more than two routes changes the usefulness of this rule'},adaptive,frozen,oracle,diagnostics:{checkpointCycle:8,at8,at16,continuedB,frozenB,freshB,retention:{beforeB:probeA(at8),afterB:probeA(at16),loss:probeA(at8)-probeA(at16)},forwardTransfer:{definition:'eight B-circle online reward, aged minus fresh under same update rule and budget',aged:continuedB.total,fresh:freshB.total,difference:continuedB.total-freshB.total},finalFrozenA:{adaptive:probeA(adaptive.finalSide),frozen:probeA(frozen.finalSide)},adaptation:{B:adaptive.circles.slice(8,16).map(c=>c.reward),returnA:adaptive.circles.slice(16).map(c=>c.reward),lostRewardPerSwitch:1},permissions:'diagnostic copies may be placed in fixed A/B and must not write their data back to the training learner'}};
}
