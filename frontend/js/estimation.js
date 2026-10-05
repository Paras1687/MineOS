'use strict';
(function(root){
 function client(request){
  const cache=new Map(),pending=new Map();let versionRequest;
  return async function(body){
   if(!versionRequest)versionRequest=request('/exploration/estimation-version').finally(()=>{versionRequest=null});
   const version=await versionRequest,key=JSON.stringify([version.version,body]);
   if(cache.has(key))return cache.get(key);
   if(!pending.has(key))pending.set(key,request('/exploration/estimate',body).then(result=>{if(result.version!==version.version)throw Error('Model updated during request. Please retry.');cache.set(key,result);if(cache.size>100)cache.delete(cache.keys().next().value);return result}).finally(()=>pending.delete(key)));
   return pending.get(key);
  };
 }
 function guard(alive){let generation=0;return {invalidate(){generation++},async run(work,success,failure){const id=++generation;try{const result=await work();if(id===generation&&alive())success(result)}catch(e){if(id===generation&&alive())failure(e)}}};}
 const e=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const n=v=>Number(v).toLocaleString('en-IN',{maximumFractionDigits:2});
 let shared;
 function attach(host,point){
  if(!host)return;
  shared ||= client((path,body)=>api(path,body));
  const target=Object.freeze({...point});let inputs={},busy=false;
  host.classList.add('estimate-host');host.innerHTML='<button type="button" class="primary estimate-open">Estimate grade &amp; reserve</button><section class="estimate-card" hidden aria-live="polite"></section>';
  const open=host.querySelector('button'),card=host.querySelector('section'),view=guard(()=>host.isConnected);
  const heading=()=>'<h3>Grade &amp; reserve evidence</h3><p><b>'+target.latitude.toFixed(6)+', '+target.longitude.toFixed(6)+'</b> · CNN index <b>'+n(target.prospectivity_score*100)+'/100</b></p>'+(target.note?'<p class="muted">'+e(target.note)+'</p>':'');
  const run=()=>{if(busy)return;busy=true;open.disabled=true;card.hidden=false;card.innerHTML=heading()+'<p role="status">Reading grade and reserve evidence…</p>';view.run(()=>shared({latitude:target.latitude,longitude:target.longitude,prospectivity_score:target.prospectivity_score,...inputs}),d=>{busy=false;open.disabled=false;draw(d)},err=>{busy=false;open.disabled=false;card.innerHTML=heading()+'<p class="error" role="alert">'+e(err.message)+'</p><button type="button" class="estimate-retry">Retry estimate</button>';card.querySelector('button').onclick=run})};
  function draw(d){
   const g=d.grade,t=d.tonnage;
   let grade=g.grade_pct!=null?n(g.grade_pct)+'% Mn':g.kind==='lower_bound'?'≥'+n(g.lower_pct)+'% Mn':g.range_pct?g.range_pct.map(n).join('–')+'% Mn':'Insufficient support';
   if(g.status==='conflicting_records')grade='Conflicting source grades';
   card.innerHTML=heading()+'<div class="estimate-values"><div><small>'+e(g.method||'Mn grade')+' · '+e(g.status.replaceAll('_',' '))+'</small><strong>'+grade+'</strong>'+(g.grade_pct!=null&&g.range_pct&&g.range_pct[0]!==g.range_pct[1]?'<p>Screening range '+g.range_pct.map(n).join('–')+'% Mn</p>':'')+'<p>'+e(g.reason)+'</p>'+ (g.recorded_values||[]).map(v=>'<p>Original: '+e(v.raw)+'</p>').join('')+'<p>'+e(g.range_basis||g.next_input)+'</p>'+ (g.quality_flags||[]).map(v=>'<p class="notice">'+e(v)+'</p>').join('')+'</div><div><small>Supplied reserve record</small>'+ (d.reserve.records.length?d.reserve.records.map(r=>'<p><b>'+e(r.original_text)+'</b><br>Unit: '+e(r.original_unit||'unspecified — no conversion')+'<br>'+e(r.name)+' · CSV row '+r.csv_row+' · '+e(r.source)+'</p>').join(''):'<p>'+e(d.reserve.reason)+'</p>')+'</div></div>'+
   '<div class="estimate-tonnage"><h4>'+e(t.label)+'</h4>'+(t.status==='calculated_from_assumptions'?'<strong>'+n(t.lower)+'–'+n(t.upper)+' '+e(t.unit)+'</strong><p>Calculated from your assumptions: area × equivalent thickness × bulk density.</p><p>Basis: '+e(t.inputs.basis)+'</p>'+(t.contained_mn?'<p>Contained Mn: '+n(t.contained_mn.lower)+'–'+n(t.contained_mn.upper)+' tonnes; no recovery or dilution applied.</p>':'')+'<p>'+e(t.limitations)+'</p>':'<p>'+e(t.reason)+'</p><p>Needed: '+e(t.missing_inputs.join(', '))+'</p>')+'</div>'+
   '<details><summary>Calculate rough tonnage scenario</summary><p>Enter mineralized geometry, not the satellite footprint or 5 km circle. No values are assumed.</p><form class="tonnage-form"><div class="estimate-inputs">'+[['area_m2','Mineralized area (m²)'],['equivalent_thickness_m','Equivalent thickness (m)'],['bulk_density_t_m3','Bulk density (t/m³)']].map(([key,label])=>'<fieldset><legend>'+label+'</legend>'+['lower','upper'].map(bound=>'<label>'+bound+'<input name="'+key+'_'+bound+'" type="number" step="any" min="0.000001" required value="'+e(inputs.scenario?.[key]?.[bound]??'')+'"></label>').join('')+'</fieldset>').join('')+'</div><label>Input basis / survey or scenario reference<input name="basis" required minlength="5" value="'+e(inputs.scenario?.basis||'')+'"></label><button type="submit">Calculate scenario</button><p class="scenario-error" role="alert"></p></form></details>'+
   (g.status==='insufficient_data'?'<details><summary>Supply local geology evidence</summary><p>Regional validation and geological support checks still apply.</p><form class="geology-form"><div class="form-grid">'+['Geology','Lithology','Host_Rock','Formation','Age','source_reference'].map(k=>'<label>'+e(k.replaceAll('_',' '))+'<input name="'+k+'" '+(['Formation','Age'].includes(k)?'':'required')+' value="'+e(inputs.geology?.[k]||'')+'"></label>').join('')+'</div><button>Check geological support</button></form></details>':'')+
   '<details><summary>Method, source &amp; limitations</summary><p>'+e(d.sources.filename)+' · '+e(d.sources.provenance)+'</p><p>CSV rows: '+e(d.sources.csv_rows.join(', ')||'No matched record')+'</p>'+(d.matched_location?'<p>Coordinate association within '+d.matched_location.distance_m+' m; not a deposit boundary.</p>':'')+'<p>'+e(d.prospectivity_basis)+'</p>'+d.limitations.map(x=>'<p>'+e(x)+'</p>').join('')+(g.evaluation?'<p>Spatial MAE '+n(g.evaluation.mae_pct)+' percentage points; median baseline '+n(g.evaluation.median_baseline_mae_pct)+'. Interval-midpoint labels.</p>':'')+'<p class="estimate-version">Version: '+e(d.version)+'<br>Data: '+e(d.data_version)+'<br>Model: '+e(d.model_version)+'</p></details>';
   card.querySelector('.tonnage-form').onsubmit=ev=>{ev.preventDefault();const values=Object.fromEntries(new FormData(ev.target)),scenario={basis:values.basis.trim()};for(const key of ['area_m2','equivalent_thickness_m','bulk_density_t_m3']){scenario[key]={lower:Number(values[key+'_lower']),upper:Number(values[key+'_upper'])};if(scenario[key].lower>scenario[key].upper){card.querySelector('.scenario-error').textContent='Upper values must be at least lower values.';return}}inputs={...inputs,scenario};run()};
   const form=card.querySelector('.geology-form');if(form)form.onsubmit=ev=>{ev.preventDefault();inputs={...inputs,geology:Object.fromEntries(new FormData(form))};run()};
  }
  open.onclick=run;
  return {invalidate(){view.invalidate();host.replaceChildren()}};
 }
 root.MineOSEstimation={attach,client,guard};
 if(typeof module!=='undefined')module.exports=root.MineOSEstimation;
})(globalThis);
