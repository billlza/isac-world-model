"""Self-contained trace viewer: stdlib only, never sends data to a service."""
from __future__ import annotations
import json
from pathlib import Path


def write_viewer(report: dict, destination: Path) -> None:
    payload = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISAC-WM v0.1 trace replay</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#fafafa;color:#202830}h1{font-size:28px}canvas{background:white;border:1px solid #ccd2d9;max-width:100%}select,input,button{font:inherit;margin:8px}pre{white-space:pre-wrap}small{display:block;line-height:1.5}#panel{display:flex;gap:24px;flex-wrap:wrap}#metrics{max-width:430px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>
<h1>ISAC-WM · v0.1 research prototype</h1>
<p>Random initialization · synthetic scene · causal sensor release</p>
<small>This viewer is an engineering smoke test, not evidence of novel or superior performance. Dashed/solid traces below show the simulated truth and an estimator, not real measurements. Predicted uncertainty is not a safety guarantee.</small>
<label>Run <select id="run"></select></label><button id="play">Play / pause</button><br>
<input id="step" type="range" min="0" value="0" style="width:80%"><span id="clock"></span>
<div id="panel"><canvas id="scene" width="520" height="520"></canvas><pre id="metrics"></pre></div>
<h2>Paired-test summary</h2><table id="summary"></table>
<h2>Scope</h2><pre id="limitations"></pre>
<script>const D=__PAYLOAD__;const rsel=document.getElementById('run'),slider=document.getElementById('step'),c=document.getElementById('scene'),ctx=c.getContext('2d');D.runs.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${r.estimator} / ${r.policy} · seed ${r.seed}`;rsel.append(o)});let playing=false;
function draw(){let r=D.runs[+rsel.value], k=Math.min(+slider.value,r.trace.length-1),v=r.trace[k],room=D.scene.room_m;slider.max=r.trace.length-1;ctx.clearRect(0,0,520,520);const p=(s)=>[30+s[0]/room*460,490-s[1]/room*460];ctx.strokeStyle='#d1d5db';ctx.strokeRect(30,30,460,460);for(const [key,col,dash] of [['truth','#374151',[5,4]],['estimate','#267a9a',[]]]){ctx.strokeStyle=col;ctx.setLineDash(dash);ctx.beginPath();r.trace.slice(0,k+1).forEach((a,i)=>{let [x,y]=p(a[key]);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.setLineDash([]);let [x,y]=p(v[key]);ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()}let [x,y]=p(v.estimate);ctx.strokeStyle='#267a9a';ctx.beginPath();ctx.ellipse(x,y,Math.min(460,2.448*Math.sqrt(v.position_variance[0])/room*460),Math.min(460,2.448*Math.sqrt(v.position_variance[1])/room*460),0,0,2*Math.PI);ctx.stroke();ctx.fillStyle='#202830';ctx.fillText('Dashed: truth   Solid: estimate   Ellipse: uncalibrated',30,18);document.getElementById('clock').textContent=`t=${v.time_s.toFixed(2)} s`;document.getElementById('metrics').textContent=JSON.stringify({error_m:v.error_m,action:['skip','quick','precise'][v.action],information_age_s:v.age_s,available_observations:v.valid,heuristic_accept:v.accepted,compute_ms:v.compute_s*1000,queue:v.queue_depth},null,2)}
rsel.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(+slider.value+1)%(+slider.max+1);draw()}},100);let table=document.getElementById('summary');table.innerHTML='<tr><th>Variant</th><th>RMSE (m)</th><th>p95 (m)</th><th>Sensing fraction</th></tr>';Object.entries(D.summary).forEach(([name,m])=>{let tr=document.createElement('tr');for(let text of [name,m.position_rmse_m.toFixed(3),m.position_p95_m.toFixed(3),m.sensing_airtime_fraction.toFixed(3)]){let td=document.createElement('td');td.textContent=text;tr.append(td)}table.append(tr)});document.getElementById('limitations').textContent=D.limitations.join('\\n');draw();</script></html>'''
    destination.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
