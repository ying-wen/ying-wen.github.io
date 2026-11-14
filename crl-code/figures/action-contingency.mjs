/**
 * Source-bounded protocols and exact EXPECTATIONS for a constructed teaching
 * example. No realized trials, animal response rates, or fitted learning curves.
 */
const probability=(value,name)=>{
  if(!Number.isFinite(value)||value<0||value>1)throw new RangeError(name+' must be in [0,1]');
  return value;
};
export function expectedOutcomeProbability(actionFraction,pAfterAction,pWithoutAction){
  probability(actionFraction,'actionFraction');
  probability(pAfterAction,'pAfterAction');
  probability(pWithoutAction,'pWithoutAction');
  return actionFraction*pAfterAction+(1-actionFraction)*pWithoutAction;
}
export function expectedOutcomeCount(actionWindows,totalWindows,pAfterAction,pWithoutAction){
  if(!Number.isInteger(totalWindows)||totalWindows<=0)throw new RangeError('totalWindows must be a positive integer');
  if(!Number.isInteger(actionWindows)||actionWindows<0||actionWindows>totalWindows)throw new RangeError('invalid actionWindows');
  return totalWindows*expectedOutcomeProbability(actionWindows/totalWindows,pAfterAction,pWithoutAction);
}
export function actionContingencyExample(){
  const totalWindows=40,windowSeconds=1;
  const schedules=[
    {id:'positive',label:'正关系',pAfterAction:.25,pWithoutAction:0},
    {id:'degraded',label:'零关系',pAfterAction:.25,pWithoutAction:.25},
    {id:'extinction',label:'停止投放',pAfterAction:0,pWithoutAction:0},
  ].map(s=>({...s,delta:s.pAfterAction-s.pWithoutAction}));
  const exposures=[10,20,30].map(actionWindows=>({
    actionWindows,noActionWindows:totalWindows-actionWindows,actionFraction:actionWindows/totalWindows,
    expectations:Object.fromEntries(schedules.map(s=>[s.id,expectedOutcomeCount(actionWindows,totalWindows,s.pAfterAction,s.pWithoutAction)])),
  }));
  return {kind:'exact',constructed:true,totalWindows,windowSeconds,
    outcomeConvention:'at most one delivery at the end of each equal window',
    probabilityOrigin:'externally specified teaching assumptions, not estimates from realized events',
    exposureOrigin:'three separate given allocations of action/no-action windows; not animal observations',
    realizedEvents:null,schedules,exposures};
}
export const hammondProtocol={
  source:'Hammond (1980), Experiment I, pp.298–300, Table 1 and Figure 1',kind:'schematic',
  action:'at least one lever press during an unsignaled 1-second window',
  noAction:'zero lever presses during the same kind of 1-second window',
  outcome:'water delivered at window end according to a programmed conditional probability',
  afterPretraining:[
    {id:'A1',sessions:14,pAfterAction:.05,pWithoutAction:0},
    {id:'B1',sessions:18,pAfterAction:.05,pWithoutAction:.05},
    {id:'A2',sessions:17,pAfterAction:.05,pWithoutAction:0},
    {id:'B2',sessions:18,pAfterAction:.05,pWithoutAction:.05},
  ],
  extinctionTest:false,measuredResponseRates:null,
};
export const yinExpressionProtocol={
  source:'Yin, Ostlund, Knowlton & Balleine (2005), Experiment 3, pp.515, 518–521, Figures 7–8',kind:'schematic',
  actions:2,outcomes:['pellets','20% sucrose solution'],assignmentCounterbalanced:true,
  training:'CRF, RR5, RR10, RR20, two days at each stage; each action trained separately',
  devaluation:{prefeedMinutes:60,infusionAfterPrefeeding:true,drugDuringAcquisition:false,
    extinctionMinutes:10,rewardedTestMinutes:20},
  degradation:{retrainingDays:1,degradationDays:3,infusionAfterDegradationLearning:true,extinctionMinutes:5},
  groups:['ACSF vehicle','muscimol into pDMS'],measuredResponseRates:null,
};
