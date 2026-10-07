'use strict';
// Frontend adapter: backend responses must carry the current selectionVersion.
const byId=id=>document.getElementById(id),map=L.map('map-container').setView([22,79],5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,className:'dark-basemap',attribution:'© OpenStreetMap contributors'}).addTo(map);
let selected=null,marker=null,scanCircle=null,selectionVersion=0;
let plotResizeObserver=null;
const pendingRequests=new Set();
function selectPoint(latitude,longitude,zoom){
 selected={latitude,longitude};selectionVersion++;
 if(marker)map.removeLayer(marker);
 marker=L.circleMarker([latitude,longitude],{radius:7,color:'#fff',weight:2,fillColor:'#58a6ff',fillOpacity:1}).addTo(map);
 if(scanCircle){map.removeLayer(scanCircle);scanCircle=null}
 if(zoom)map.setView([latitude,longitude],12);
 byId('val-coords').textContent='Lat: '+latitude.toFixed(5)+', Lon: '+longitude.toFixed(5);
 for(const id of ['val-gravity','val-prob'])byId(id).textContent='-';
 byId('model-inputs').hidden=true;byId('band-values').replaceChildren();
 byId('plot-status').textContent='Fetching satellite bands and gravity…';drawPreview(null);
 byId('val-status').textContent='Point selected';byId('scan-nearby').disabled=false;
 byId('scan-status').textContent='';byId('priority-results').hidden=true;byId('priority-list').replaceChildren();
 window.dispatchEvent(new CustomEvent('mineos:point-selected',{detail:{...selected,selectionVersion}}));
}
map.on('click',ev=>{byId('mine-select').value='';selectPoint(ev.latlng.lat,ev.latlng.lng,false)});
async function loadMines(){
 try{const r=await fetch('/api/v1/mines');if(!r.ok)throw Error();const data=await r.json();
 const mines=data.data.filter(m=>m.country==='India'&&Number.isFinite(m.latitude)&&Number.isFinite(m.longitude));
 mines.sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));
 for(const m of mines){const option=document.createElement('option');option.value=m.id;option.textContent=m.name;byId('mine-select').append(option)}
 byId('mine-select').onchange=()=>{const mine=mines.find(m=>m.id===byId('mine-select').value);if(mine)selectPoint(mine.latitude,mine.longitude,true)};
 }catch{byId('mine-select').options[0].textContent='Mine list unavailable';}
}
byId('scan-nearby').onclick=()=>{
 if(!selected)return;
 if(scanCircle)map.removeLayer(scanCircle);
 scanCircle=L.circle([selected.latitude,selected.longitude],{radius:5000,color:'#58a6ff',weight:2,fillOpacity:.07}).addTo(map);
 map.fitBounds(scanCircle.getBounds(),{padding:[20,20]});
 byId('scan-status').textContent='Scanning satellite patches within 5 km…';
 window.dispatchEvent(new CustomEvent('mineos:scan-requested',{detail:{...selected,radius_km:5,selectionVersion}}));
};
function setNearbyResults(features,version){
 if(version!==selectionVersion||!selected)return false;
 const origin=L.latLng(selected.latitude,selected.longitude);
 const results=features.map(f=>f.properties||f).filter(p=>Number.isFinite(p.score)&&p.score>=.9&&p.score<=1&&Number.isFinite(p.latitude)&&Math.abs(p.latitude)<=90&&Number.isFinite(p.longitude)&&Math.abs(p.longitude)<=180).map(p=>({...p,distance_km:origin.distanceTo([p.latitude,p.longitude])/1000})).filter(p=>p.distance_km<=5).sort((a,b)=>b.score-a.score||a.distance_km-b.distance_km);
 byId('priority-list').replaceChildren();byId('priority-results').hidden=false;
 results.forEach((p,i)=>{const li=document.createElement('li'),button=document.createElement('button'),rank=document.createElement('b'),detail=document.createElement('small');
 rank.textContent='Priority '+(i+1)+' · '+(p.score*100).toFixed(1)+'/100';detail.textContent=p.latitude.toFixed(5)+', '+p.longitude.toFixed(5)+' · '+p.distance_km.toFixed(2)+' km';button.append(rank,detail);button.onclick=()=>{map.setView([p.latitude,p.longitude],14);L.popup().setLatLng([p.latitude,p.longitude]).setContent(rank.textContent).openOn(map)};li.append(button);byId('priority-list').append(li)});
 byId('scan-status').textContent=results.length?results.length+' high-score locations, ranked by score then distance.':'No 90+ results returned within 5 km.';return true;
}
function setPointResult(result,version){
 if(version!==selectionVersion||!selected)return false;
 byId('val-gravity').textContent=Number.isFinite(result.gravity_mgal)?result.gravity_mgal.toFixed(2)+' mGal':'-';
 byId('val-prob').textContent=Number.isFinite(result.score)?(result.score*100).toFixed(1)+'/100':'-';
 byId('val-status').textContent=result.status||'-';
 byId('val-prob').title=result.score_meaning||result.score_status||'';
 const patchWidth=Number.isFinite(result.patch_width_m)?result.patch_width_m:640;
 byId('plot-status').textContent=Number.isFinite(result.score)?'Patch score '+(result.score*100).toFixed(1)+'/100 across a '+patchWidth+' m image footprint. Subsurface depth is not estimated.':'No score available for this point.';
 drawPreview(result);
 if(Array.isArray(result.input_band_order)&&Array.isArray(result.input_band_medians)){
  byId('band-values').replaceChildren();
  result.input_band_order.forEach((name,i)=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=name;dd.textContent=Number(result.input_band_medians[i]).toFixed(4)+(i<6?' reflectance':'');byId('band-values').append(dt,dd)});
  if(Number.isFinite(result.input_valid_fraction)){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent='Clear imagery';dd.textContent=(result.input_valid_fraction*100).toFixed(1)+'%';byId('band-values').append(dt,dd)}
  byId('model-inputs').hidden=false;
 }
 return true;
}
async function submitExploration(path,detail,nearby){
 const version=detail.selectionVersion;
 const requestKey=path+':'+version;
 if(pendingRequests.has(requestKey))return;
 pendingRequests.add(requestKey);
 if(nearby)byId('scan-nearby').disabled=true;
 if(!nearby)byId('val-status').textContent='Fetching satellite imagery and scoring…';
 try{
  const response=await fetch('/api/v1/exploration/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({latitude:detail.latitude,longitude:detail.longitude,cloud:true}),signal:AbortSignal.timeout(60000)});
  if(!response.ok)throw Error((await response.json()).detail||'Could not start model job');
  const {job_id}=await response.json();
  for(let i=0;i<360;i++){
   if(version!==selectionVersion)return;
   await new Promise(resolve=>setTimeout(resolve,1500));
   const poll=await fetch('/api/v1/jobs/'+job_id,{signal:AbortSignal.timeout(30000)});
   if(poll.status===502||poll.status===503||poll.status===504)continue;
   if(poll.status===404)throw Error('Server restarted and this request expired. Select the point again to retry.');
   const job=await poll.json();
   if(!poll.ok)throw Error(job.detail||'Model job status could not be read');
   if(job.status==='complete'){
    if(nearby){if(version===selectionVersion&&selected){setNearbyResults(job.result.features||[],version);byId('scan-status').textContent=job.result.message||byId('scan-status').textContent}}
    else setPointResult(job.result,version);
    return;
   }
   if(job.status==='failed')throw Error(job.error||'Model could not score this point');
  }
  throw Error('Model request timed out; retry by inspecting or scanning again');
 }catch(error){
  if(version!==selectionVersion)return;
  if(nearby)byId('scan-status').textContent=error.message;
  else {byId('val-status').textContent=error.message;byId('plot-status').textContent='Prediction unavailable. Select the point again to retry.';}
 }finally{
  pendingRequests.delete(requestKey);
  if(nearby&&version===selectionVersion)byId('scan-nearby').disabled=false;
 }
}
window.addEventListener('mineos:point-selected',event=>submitExploration('predict',event.detail,false));
window.addEventListener('mineos:scan-requested',event=>submitExploration('nearby',event.detail,true));
function drawPreview(result=null){
 if(!window.Plotly){byId('plot-container').textContent='3D viewer could not load.';return}
 const score=typeof result==='number'?result:result?.score;
 const width=typeof result==='object'&&Number.isFinite(result?.patch_width_m)?result.patch_width_m:640;
 const half=width/2,data=[];
 if(Number.isFinite(score)){
  const s=Math.max(0,Math.min(100,score*100));
  const x=[-half,half,half,-half,-half,half,half,-half],y=[-half,-half,half,half,-half,-half,half,half],z=[0,0,0,0,s,s,s,s];
  data.push({type:'mesh3d',x,y,z,i:[0,0,4,4,0,0,1,1,2,2,3,3],j:[2,3,5,6,1,5,2,6,3,7,0,4],k:[1,2,6,7,5,4,6,5,7,6,4,7],intensity:[0,0,0,0,s,s,s,s],intensitymode:'vertex',colorscale:[[0,'#23405b'],[.55,'#d2a73c'],[1,'#3fb950']],cmin:0,cmax:100,showscale:false,opacity:.86,hovertemplate:'Patch score: '+s.toFixed(1)+'/100<extra>Model output</extra>',name:'Patch score'});
 }
 Plotly.react('plot-container',data,{paper_bgcolor:'#0d1117',font:{color:'#8b949e'},scene:{xaxis:{title:{text:'East offset (m)'},range:[-half,half],gridcolor:'#30363d',zeroline:false},yaxis:{title:{text:'North offset (m)'},range:[-half,half],gridcolor:'#30363d',zeroline:false},zaxis:{title:{text:'Prospectivity score (0–100)'},range:[0,100],dtick:20,gridcolor:'#30363d'},aspectmode:'manual',aspectratio:{x:1,y:1,z:.72},camera:{eye:{x:1.55,y:1.55,z:.85}}},margin:{l:0,r:0,b:8,t:0}},{responsive:true,displayModeBar:false,displaylogo:false});
 if(!plotResizeObserver){plotResizeObserver=new ResizeObserver(()=>Plotly.Plots.resize('plot-container'));plotResizeObserver.observe(byId('plot-container'));}
}
window.MineOSExplorer={setNearbyResults,setPointResult,getSelection:()=>selected?{...selected,selectionVersion}:null};
loadMines();drawPreview();
