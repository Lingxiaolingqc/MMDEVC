'use strict';
const $ = id => document.getElementById(id);
const colors = ['#6dafff','#63dfb1','#f5bf64','#dd8cf5','#fb8c8c','#8ce5ef'];
let meta, pointData, frame = 0, method = 0, x = 0, y = 0, generation = 0, pointGeneration = 0;
let rgbImage, depthImage, playing = false, startClock = 0, startTime = 0;
const status = text => { $('status').textContent = text; };
async function json(url) { const response = await fetch(url); if (!response.ok) throw new Error(`读取失败 (${response.status})`); return response.json(); }
async function bitmap(index, id) { const r = await fetch(`/api/frame?index=${index}&method=${id}`); if (!r.ok) throw new Error('无法读取画面'); return createImageBitmap(await r.blob()); }
function drawImage(canvas, image) {
  const c = canvas.getContext('2d'); c.clearRect(0,0,canvas.width,canvas.height); if (image) c.drawImage(image,0,0);
  c.strokeStyle = '#fff'; c.lineWidth = Math.max(1,canvas.width/360); c.shadowColor = '#000'; c.shadowBlur = 2;
  c.beginPath(); c.arc(x+.5,y+.5,Math.max(4,canvas.width/90),0,Math.PI*2); c.stroke(); c.shadowBlur = 0;
}
function drawChart() {
  if (!pointData) return;
  const canvas=$('chart'), rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*ratio); canvas.height=Math.round(rect.height*ratio);
  const c=canvas.getContext('2d');c.scale(ratio,ratio);const W=rect.width,H=rect.height,L=48,R=20,T=24,B=30;
  const all=pointData.traces.flatMap(t=>t.values.filter(v=>v!==null));
  const ymax=Math.max(1,Math.ceil(Math.max(0,...all)*1.1)), first=meta.timestamps[0], last=meta.timestamps.at(-1);
  const X=t=>L+(t-first)/Math.max(.001,last-first)*(W-L-R), Y=v=>H-B-v/ymax*(H-T-B);
  c.font='11px system-ui';c.lineWidth=1;
  for(let k=0;k<=4;k++){const v=ymax*k/4;c.strokeStyle='#26364c';c.beginPath();c.moveTo(L,Y(v));c.lineTo(W-R,Y(v));c.stroke();c.fillStyle='#8da4c1';c.fillText(v.toFixed(1),8,Y(v)+4);}
  c.fillText('m',8,12);c.fillText(`${first.toFixed(1)} s`,L,H-9);c.fillText(`${last.toFixed(2)} s`,W-R-42,H-9);
  pointData.traces.forEach((trace,i)=>{c.strokeStyle=colors[i%colors.length];c.lineWidth=i===method?2.7:1.5;c.beginPath();let connected=false;
    trace.values.forEach((v,t)=>{if(v===null){connected=false;return;} if(connected)c.lineTo(X(meta.timestamps[t]),Y(v));else c.moveTo(X(meta.timestamps[t]),Y(v));connected=true;});c.stroke();});
  c.strokeStyle='#e7edf7';c.setLineDash([3,3]);c.beginPath();c.moveTo(X(meta.timestamps[frame]),T);c.lineTo(X(meta.timestamps[frame]),H-B);c.stroke();c.setLineDash([]);
}
function readout() {
  $('point').textContent=`(${x}, ${y})`;
  const value=pointData?.x===x && pointData?.y===y ? pointData.traces[method]?.values[frame] : undefined;
  $('value').textContent=value===null?'无有效预测':value===undefined?'…':`${value.toFixed(3)} m`;
  $('time').textContent=`${meta.timestamps[frame].toFixed(3)} s · ${frame+1}/${meta.frames}`;
  $('depth-title').textContent=meta.methods[method].label;
  drawChart();
}
async function showFrame(index) {
  frame=Math.max(0,Math.min(meta.frames-1,index));$('frame').value=frame;
  const request=++generation, selected=frame, id=method;
  try { const [rgb,depth]=await Promise.all([bitmap(selected,-1),bitmap(selected,id)]);
    if(request!==generation){rgb.close();depth.close();return;}
    rgbImage?.close();depthImage?.close();rgbImage=rgb;depthImage=depth;
    drawImage($('rgb'),rgbImage);drawImage($('depth'),depthImage);readout();status('点击任一画面可更换位置。');
  } catch(error){if(request===generation)status(error.message);}
}
async function selectPoint(event) {
  const rect=event.currentTarget.getBoundingClientRect();
  x=Math.max(0,Math.min(meta.width-1,Math.floor((event.clientX-rect.left)/rect.width*meta.width)));
  y=Math.max(0,Math.min(meta.height-1,Math.floor((event.clientY-rect.top)/rect.height*meta.height)));
  const request=++pointGeneration;
  try {const values=await json(`/api/point?x=${x}&y=${y}`);if(request!==pointGeneration)return;pointData=values;
    drawImage($('rgb'),rgbImage);drawImage($('depth'),depthImage);readout();
  } catch(error){status(error.message);}
}
function stop(){playing=false;$('play').textContent='播放';}
function tick(now){
  if(!playing)return;
  const t=startTime+(now-startClock)/1000;let next=frame;
  while(next+1<meta.frames&&meta.timestamps[next+1]<=t)next++;
  if(next!==frame)showFrame(next);
  if(t>=meta.timestamps.at(-1)){stop();return;}requestAnimationFrame(tick);
}
async function init(){
  try {meta=await json('/api/meta');x=Math.floor(meta.width/2);y=Math.floor(meta.height/2);
    method=Math.max(0,meta.methods.findIndex(m=>m.name==='E4'));
    $('clip').textContent=`${meta.clip_id} · ${meta.width} × ${meta.height} · ${meta.frames} 帧`;
    const notices=[];if(meta.scope!=='self_recorded_project')notices.push('示例素材：用于验证程序，不属于正式自录实验。');
    if(meta.methods.some(m=>m.provisional))notices.push('融合参数仍待开发集验证。');
    if(notices.length){$('notice').hidden=false;$('notice').textContent=notices.join(' ');}
    for(const canvas of [$('rgb'),$('depth')]){canvas.width=meta.width;canvas.height=meta.height;canvas.addEventListener('click',selectPoint);}
    meta.methods.forEach((m,i)=>{const option=document.createElement('option');option.value=i;option.textContent=m.label;$('method').append(option);
      const label=document.createElement('span');label.textContent=m.name;label.style.setProperty('--color',colors[i%colors.length]);$('legend').append(label);});
    $('method').value=method;$('depth-max').textContent=`${meta.depth_max} m`;
    $('method').addEventListener('change',()=>{method=Number($('method').value);showFrame(frame);});
    $('frame').max=meta.frames-1;$('frame').disabled=false;$('frame').addEventListener('input',()=>{stop();showFrame(Number($('frame').value));});
    $('play').disabled=false;$('play').addEventListener('click',()=>{if(playing){stop();return;}if(frame===meta.frames-1)showFrame(0);playing=true;$('play').textContent='暂停';startClock=performance.now();startTime=meta.timestamps[frame];requestAnimationFrame(tick);});
    pointData=await json(`/api/point?x=${x}&y=${y}`);await showFrame(0);window.addEventListener('resize',drawChart);
  }catch(error){status(error.message);}
}
init();
