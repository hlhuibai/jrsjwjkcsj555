"""
Web展示层：Flask路由 + HTML仪表板
仅供前端同学修改，策略逻辑通过 engine 模块调用
"""

import json, os, sys
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

sys.path.insert(0, os.path.dirname(__file__))
from engine import (
    FULL_UNIVERSE, BENCHMARK_INDICES, DEFAULT_CONFIG,
    TRAIN_PERIOD, TEST_PERIOD, TEST_TRADE_START,
    generate_price_data, generate_index_data,
    run_dual_period, _data_cache, _load_cache,
)
from indicators import (
    calc_main_force_score, calc_institutional_flow, calc_mfi,
    detect_accumulation,
)

app = Flask(__name__)
_cached_result = None


# ═══════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════

@app.route("/echarts.min.js")
def echarts_js():
    return send_file(os.path.join(os.path.dirname(__file__), "echarts.min.js"))

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    global _cached_result
    config = request.get_json(silent=True) or {}
    params = {k:v for k,v in config.items() if k in DEFAULT_CONFIG}
    _cached_result = run_dual_period(params)
    return jsonify({"ok":True,"data":_cached_result})

@app.route("/api/backtest", methods=["GET"])
def api_get_result():
    global _cached_result
    if _cached_result is None:
        _cached_result = run_dual_period()
    return jsonify({"ok":True,"data":_cached_result})

@app.route("/api/stock/<symbol>")
def api_stock_detail(symbol):
    df = generate_price_data(symbol, TRAIN_PERIOD[0], TRAIN_PERIOD[1])
    if df is None: return jsonify({"ok":False,"error":"数据不足"}),404
    high, low, close, volume = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    scores = calc_main_force_score(high, low, close, volume, 14)
    mfi_vals = calc_mfi(high, low, close, volume, 14)
    flow_vals = calc_institutional_flow(high, low, close, volume)
    acc = detect_accumulation(close, volume, 20)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    # 获取该股的买卖点
    buy_pts, sell_pts = [], []
    if _cached_result:
        for period in ["train","test"]:
            for t in _cached_result.get(period,{}).get("trades",[]):
                if t["symbol"] == symbol:
                    pt = [t["date"], t["price"]]
                    if t["action"] == "buy": buy_pts.append(pt)
                    else: sell_pts.append(pt)
    return jsonify({"ok":True,"data":{
        "symbol":symbol,"dates":dates,
        "candles":[[float(o),float(c),float(lo),float(h),int(v)]
                   for o,c,lo,h,v in zip(df["open"].values,close,low,high,volume)],
        "scores":[float(s) if not np.isnan(s) else None for s in scores],
        "mfi":[float(m) if not np.isnan(m) else None for m in mfi_vals],
        "flow":[float(f) if not np.isnan(f) else None for f in flow_vals],
        "accumulation":[float(a) for a in acc],
        "buy_points": buy_pts,
        "sell_points": sell_pts,
    }})

@app.route("/api/stocks")
def api_stocks():
    return jsonify({"ok":True,"data":[{"symbol":s,"name":n} for s,n in FULL_UNIVERSE]})


# ═══════════════════════════════════════════════════════════
# HTML 模板
# ═══════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>涨停放量策略 - 回测仪表板</title>
<script src="/echarts.min.js"></script>
<style>
:root{--bg:#0f1923;--card-bg:#1a2332;--border:#2a3a4a;--text:#c8d6e5;--text-dim:#6b7d8e;--green:#00d4aa;--red:#ff4757;--blue:#4a9eff;--accent:#f9ca24}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
header{background:var(--card-bg);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100}
header h1{font-size:20px;font-weight:600}
header .controls{display:flex;gap:8px;align-items:center}
header button{padding:8px 20px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;transition:all .2s}
.btn-run{background:var(--blue);color:#fff}.btn-run:hover{background:#3a8aee}.btn-run:disabled{opacity:.5;cursor:not-allowed}
.btn-config{background:var(--border);color:var(--text)}.btn-config:hover{background:#3a4a5a}
main{padding:20px 24px;max-width:1600px;margin:0 auto}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:20px}
.metric-card{background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:16px 20px;text-align:center}
.metric-card .label{font-size:12px;color:var(--text-dim);margin-bottom:6px}
.metric-card .value{font-size:18px;font-weight:700}
.metric-card .value.green{color:var(--green)}.metric-card .value.red{color:var(--red)}.metric-card .value.blue{color:var(--blue)}
.metric-card .sub{font-size:10px;color:var(--text-dim);margin-top:2px}
.chart-grid{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-bottom:20px}
.chart-box{background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:14px}
.chart-box h3{font-size:14px;font-weight:500;margin-bottom:10px;color:var(--text-dim)}
.chart{width:100%;height:380px}
.stock-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
.table-box{background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:20px;max-height:400px;overflow-y:auto}
.table-box h3{font-size:14px;margin-bottom:10px;color:var(--text-dim);position:sticky;top:0;background:var(--card-bg);padding-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--text-dim);padding:8px 6px;border-bottom:1px solid var(--border);position:sticky;top:36px;background:var(--card-bg)}
td{padding:6px;border-bottom:1px solid rgba(42,58,74,.4)}
td.green{color:var(--green)}td.red{color:var(--red)}
tr:hover td{background:rgba(74,158,255,.05)}
.modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);z-index:200;justify-content:center;align-items:center}
.modal-overlay.active{display:flex}
.modal{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:24px;width:480px;max-height:80vh;overflow-y:auto}
.modal h2{font-size:18px;margin-bottom:16px}
.modal .field{margin-bottom:14px}
.modal label{display:block;font-size:12px;color:var(--text-dim);margin-bottom:4px}
.modal input,.modal select{width:100%;padding:8px 12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:14px}
.modal .field-row{display:flex;gap:12px}
.modal .field-row .field{flex:1}
.modal .actions{display:flex;gap:8px;justify-content:flex-end;margin-top:20px}
.spinner{display:none;width:28px;height:28px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite}
.spinner.active{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px}
.badge-buy{background:rgba(0,212,170,.15);color:var(--green)}.badge-sell{background:rgba(255,71,87,.15);color:var(--red)}
@media(max-width:1100px){.chart-grid,.stock-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>涨停放量策略 · 回测仪表板</h1>
  <div class="controls">
    <div class="spinner" id="spinner"></div>
    <button class="btn-config" onclick="openConfig()">参数配置</button>
    <button class="btn-run" id="btnRun" onclick="runBacktest()">运行回测</button>
  </div>
</header>
<main>
  <div class="metrics-grid" id="metricsGrid"></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px">
    <div class="table-box" style="max-height:200px">
      <h3>训练集持仓 <span id="trainHoldingVal" style="font-size:12px;color:var(--text-dim)"></span></h3>
      <table><thead><tr><th>股票</th><th>手</th><th>成本</th><th>现价</th><th>浮盈%</th></tr></thead><tbody id="trainHoldingBody"><tr><td colspan="5" style="text-align:center;color:var(--text-dim)">暂无</td></tr></tbody></table>
    </div>
    <div class="table-box" style="max-height:200px">
      <h3>测试集持仓 <span id="testHoldingVal" style="font-size:12px;color:var(--text-dim)"></span></h3>
      <table><thead><tr><th>股票</th><th>手</th><th>成本</th><th>现价</th><th>浮盈%</th></tr></thead><tbody id="testHoldingBody"><tr><td colspan="5" style="text-align:center;color:var(--text-dim)">暂无</td></tr></tbody></table>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-box"><h3>净值曲线</h3><div class="chart" id="chartEquity"></div></div>
    <div class="chart-box"><h3>回撤曲线</h3><div class="chart" id="chartDrawdown"></div></div>
  </div>
  <div class="stock-grid">
    <div class="chart-box">
      <h3>个股K线 + 买卖点</h3>
      <select id="stockSelect" onchange="loadStockDetail()" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 12px;"></select>
      <div class="chart" id="chartStock" style="height:450px;"></div>
    </div>
  </div>
  <div class="table-box">
    <h3>交易记录（最近50笔）</h3>
    <table><thead><tr>
      <th>日期</th><th>股票</th><th>方向</th><th>价格</th><th>股数</th><th>金额</th><th>盈亏%</th><th>原因</th>
    </tr></thead><tbody id="tradeBody"></tbody></table>
  </div>
</main>
<div class="modal-overlay" id="configModal">
  <div class="modal">
    <h2>回测参数配置</h2>
    <div class="field-row">
      <div class="field"><label>训练集</label><input value="2024-01-01 ~ 2025-12-31" readonly style="background:var(--bg);color:var(--text-dim);"></div>
      <div class="field"><label>测试集</label><input value="2026-01-01 ~ 2026-05-27" readonly style="background:var(--bg);color:var(--text-dim);"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>初始资金</label><input type="number" id="cfgCapital" value="1000000" step="100000"></div>
      <div class="field"><label>最大持仓数</label><input type="number" id="cfgMaxPos" value="8" min="1" max="20"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>底部阈值 (%)</label><input type="number" id="cfgBottomPct" value="20" min="5" max="40" step="5"></div>
      <div class="field"><label>单票最大仓位 (%)</label><input type="number" id="cfgMaxPct" value="20" min="5" max="40" step="5"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>止损 (%)</label><input type="number" id="cfgStopLoss" value="-8" min="-30" max="0" step="1"></div>
      <div class="field"><label>止盈 (%)</label><input type="number" id="cfgTakeProfit" value="20" min="5" max="100" step="5"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>保留现金 (%)</label><input type="number" id="cfgReserveCash" value="10" min="0" max="30" step="5"></div>
      <div class="field"></div>
    </div>
    <div class="actions">
      <button class="btn-config" onclick="closeConfig()">取消</button>
      <button class="btn-run" onclick="runBacktest();closeConfig()">运行</button>
    </div>
  </div>
</div>
<script>
let resultData=null;
let charts={};
document.addEventListener("DOMContentLoaded",()=>{initCharts();loadStockList();fetchCached()});
window.addEventListener("resize",()=>Object.values(charts).forEach(c=>c?.resize()));
function initCharts(){
  charts.equity=echarts.init(document.getElementById("chartEquity"));
  charts.drawdown=echarts.init(document.getElementById("chartDrawdown"));
  charts.stock=echarts.init(document.getElementById("chartStock"));
}
async function fetchCached(){
  try{const r=await fetch("/api/backtest");const j=await r.json();if(j.ok&&j.data){resultData=j.data;renderAll()}}catch(e){}
}
async function runBacktest(){
  document.getElementById("btnRun").disabled=true;
  document.getElementById("spinner").classList.add("active");
  const c={
    initial_capital:parseInt(document.getElementById("cfgCapital")?.value||1e6),
    bottom_pct:parseFloat(document.getElementById("cfgBottomPct")?.value||20)/100,
    stop_loss:parseFloat(document.getElementById("cfgStopLoss")?.value||-8)/100,
    take_profit:parseFloat(document.getElementById("cfgTakeProfit")?.value||20)/100,
    max_positions:parseInt(document.getElementById("cfgMaxPos")?.value||8),
    max_single_position:parseFloat(document.getElementById("cfgMaxPct")?.value||20)/100,
    reserve_cash_pct:parseFloat(document.getElementById("cfgReserveCash")?.value||10)/100,
  };
  try{const r=await fetch("/api/backtest",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(c)});const j=await r.json();if(j.ok){resultData=j.data;renderAll()}}catch(e){}
  finally{document.getElementById("btnRun").disabled=false;document.getElementById("spinner").classList.remove("active")}
}
async function loadStockList(){
  const r=await fetch("/api/stocks");const j=await r.json();
  const s=document.getElementById("stockSelect");
  j.data.forEach(x=>{const o=document.createElement("option");o.value=x.symbol;o.textContent=x.name+" ("+x.symbol+")";s.appendChild(o)});
}
async function loadStockDetail(){
  const sym=document.getElementById("stockSelect").value;if(!sym)return;
  const r=await fetch("/api/stock/"+sym);const j=await r.json();
  if(j.ok)renderStockChart(j.data);
}
function renderAll(){
  if(!resultData)return;
  try{
    const train=resultData.train,test=resultData.test;
    if(!train||!test)return;
    const tl=train.equity_series.length>0?train.equity_series[train.equity_series.length-1][1]:1;
    const tEq=test.equity_series.map(e=>[e[0],e[1]*tl]);
    const tDd=test.drawdown_series.map(d=>[d[0],d[1]]);
    const eq=[...train.equity_series,...tEq];
    const dd=[...train.drawdown_series,...tDd];
    const bm={};
    for(const k of Object.keys(train.benchmark_series||{})){
      bm[k]=[...train.benchmark_series[k],...(test.benchmark_series[k]||[])];
    }
    renderMetrics(train.summary,test.summary);
    renderEquityChart(eq,dd,bm,train.equity_series.length);
    renderTrades(test.trades,train.trades);
    renderHoldings(train.holdings,test.holdings);
    }catch(e){console.error(e)}
}
function renderMetrics(train,test){
  const card=(l,v1,v2,c,u)=>
    `<div class="metric-card"><div class="label">${l}</div>
      <div style="display:flex;justify-content:space-around;align-items:baseline;margin-top:4px">
        <div><div class="sub">训练集</div><div class="value ${c}" style="font-size:18px">${v1}${u||''}</div></div>
        <div style="color:var(--border);font-size:20px">|</div>
        <div><div class="sub">测试集</div><div class="value ${c}" style="font-size:18px">${v2}${u||''}</div></div>
      </div></div>`;
  document.getElementById("metricsGrid").innerHTML=
    card("总收益率",(train.total_return>0?'+':'')+train.total_return.toFixed(1)+'%',(test.total_return>0?'+':'')+test.total_return.toFixed(1)+'%',train.total_return>0?'green':'red')+
    card("年化收益",train.annual_return.toFixed(1)+'%',test.annual_return.toFixed(1)+'%',train.annual_return>0?'green':'red')+
    card("最大回撤",train.max_drawdown.toFixed(1)+'%',test.max_drawdown.toFixed(1)+'%','red')+
    card("夏普比率",train.sharpe_ratio,test.sharpe_ratio,train.sharpe_ratio>1?'green':'blue')+
    card("胜率",train.win_rate+'%',test.win_rate+'%',train.win_rate>50?'green':'blue')+
    card("交易次数",train.total_trades,test.total_trades,'blue')+
    card("手续费",(train.total_commission||0).toLocaleString(),(test.total_commission||0).toLocaleString(),'','元');
}
function renderEquityChart(equity,drawdown,benchmarks,testStart){
  // time轴按真实日历对齐，消除节假日偏差
  const eqData=equity.map(e=>[e[0],e[1]]),ddData=drawdown.map(d=>[d[0],d[1]]);
  const benchColors={"沪深300":"#f9ca24","创业板指":"#ff6b6b","科创50":"#00d4aa"};
  const bs=Object.entries(benchmarks||{}).map(([n,s])=>({type:"line",name:n,data:s.filter(x=>x&&x[1]!=null).map(x=>[x[0],x[1]]),smooth:true,symbol:"none",lineStyle:{color:benchColors[n]||"#888",width:1.5,type:"dashed"}}));
  charts.equity.setOption({
    tooltip:{trigger:"axis",backgroundColor:"rgba(26,35,50,0.95)",borderColor:"#2a3a4a",textStyle:{color:"#c8d6e5",fontSize:12}},
    legend:{data:["策略净值",...Object.keys(benchmarks||{})],bottom:0,textStyle:{color:"#6b7d8e",fontSize:10}},
    grid:{left:60,right:30,top:10,bottom:40},
    xAxis:{type:"time",axisLine:{lineStyle:{color:"#2a3a4a"}},axisLabel:{color:"#6b7d8e",fontSize:10},splitLine:{show:false}},
    yAxis:{type:"value",axisLabel:{color:"#6b7d8e",fontSize:10,formatter:v=>v.toFixed(2)},splitLine:{lineStyle:{color:"rgba(42,58,74,0.3)"}},scale:true},
    series:[{type:"line",name:"策略净值",data:eqData,smooth:true,symbol:"none",lineStyle:{color:"#4a9eff",width:2.5},
      areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:"rgba(74,158,255,0.18)"},{offset:1,color:"rgba(74,158,255,0.0)"}])},
      markLine:{silent:true,symbol:"none",data:[
        {yAxis:1,lineStyle:{color:"#6b7d8e",type:"dashed"},label:{color:"#6b7d8e",fontSize:10,formatter:"基准 1.0"}},
      ]},
    },...bs],
  },true);
  charts.drawdown.setOption({
    tooltip:{trigger:"axis",backgroundColor:"rgba(26,35,50,0.95)",borderColor:"#2a3a4a",textStyle:{color:"#c8d6e5",fontSize:12}},
    grid:{left:60,right:20,top:10,bottom:30},
    xAxis:{type:"time",axisLine:{lineStyle:{color:"#2a3a4a"}},axisLabel:{color:"#6b7d8e",fontSize:10},splitLine:{show:false}},
    yAxis:{type:"value",axisLabel:{color:"#6b7d8e",fontSize:10,formatter:v=>v+"%"},splitLine:{lineStyle:{color:"rgba(42,58,74,0.3)"}}},
    series:[{type:"line",data:ddData,symbol:"none",lineStyle:{color:"#ff4757",width:1.5},areaStyle:{color:"rgba(255,71,87,0.15)"}}],
  },true);
}
function renderTrades(testTrades,trainTrades){
  const all=[...(trainTrades||[]).map(t=>({...t,period:'训练'})),...(testTrades||[]).map(t=>({...t,period:'测试'}))];
  document.getElementById("tradeBody").innerHTML=all.slice().reverse().map(t=>`
    <tr><td>${t.date}<span style="font-size:9px;color:var(--text-dim);margin-left:2px">${t.period}</span></td>
    <td><b>${t.name||t.symbol}</b></td>
    <td><span class="badge ${t.action==='buy'?'badge-buy':'badge-sell'}">${t.action==='buy'?'买入':'卖出'}</span></td>
    <td>${t.price}</td><td>${(t.shares/100).toFixed(0)}手</td><td>${(t.amount/10000).toFixed(1)}万</td>
    <td class="${t.pnl_pct>0?'green':t.pnl_pct<0?'red':''}">${t.pnl_pct>0?'+':''}${t.pnl_pct}%</td>
    <td style="color:var(--text-dim);font-size:12px">${t.reason}</td></tr>`).join("");
}
function renderStockChart(data){
  const dates=data.dates,candles=data.candles.map((c,i)=>[c[0],c[1],c[2],c[3],c[4]]);
  const buyMarks=[],sellMarks=[];
  (data.buy_points||[]).forEach(p=>buyMarks.push({coord:[p[0],p[1]],symbol:'triangle',symbolSize:14,itemStyle:{color:'#ff0000'},label:{show:true,position:'bottom',formatter:'B',color:'#ff0000',fontSize:11,fontWeight:'bold'}}));
  (data.sell_points||[]).forEach(p=>sellMarks.push({coord:[p[0],p[1]],symbol:'triangle',symbolSize:14,symbolRotate:180,itemStyle:{color:'#00aa00'},label:{show:true,position:'top',formatter:'S',color:'#00aa00',fontSize:11,fontWeight:'bold'}}));
  charts.stock.setOption({
    tooltip:{trigger:"axis",backgroundColor:"rgba(26,35,50,0.95)",borderColor:"#2a3a4a",textStyle:{color:"#c8d6e5",fontSize:12}},
    grid:[{left:70,right:20,top:10,height:"55%"},{left:70,right:20,top:"72%",height:"23%"}],
    xAxis:[{type:"category",data:dates,gridIndex:0,axisLabel:{color:"#6b7d8e",fontSize:9,formatter:v=>v.slice(5)},axisLine:{lineStyle:{color:"#2a3a4a"}}},
           {type:"category",data:dates,gridIndex:1,axisLabel:{show:false},axisLine:{show:false},axisTick:{show:false}}],
    yAxis:[{type:"value",gridIndex:0,axisLabel:{color:"#6b7d8e",fontSize:10},splitLine:{lineStyle:{color:"rgba(42,58,74,0.3)"}},scale:true},
           {type:"value",gridIndex:1,min:0,max:100,axisLabel:{color:"#6b7d8e",fontSize:9,formatter:v=>v.toFixed(0)},splitLine:{lineStyle:{color:"rgba(42,58,74,0.2)"}}}],
    series:[
      {type:"candlestick",name:"K线",data:candles,xAxisIndex:0,yAxisIndex:0,
       itemStyle:{color:"#ff0000",color0:"#00aa00",borderColor:"#ff0000",borderColor0:"#00aa00"}},
      {type:"scatter",name:"买入",data:buyMarks,xAxisIndex:0,yAxisIndex:0,z:10},
      {type:"scatter",name:"卖出",data:sellMarks,xAxisIndex:0,yAxisIndex:0,z:10},
      {type:"line",name:"主力评分",data:data.scores,xAxisIndex:1,yAxisIndex:1,
       symbol:"none",smooth:true,lineStyle:{color:"#f9ca24",width:2},
       areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:"rgba(249,202,36,0.2)"},{offset:1,color:"rgba(249,202,36,0.0)"}])},
       markLine:{silent:true,symbol:"none",data:[{yAxis:40,lineStyle:{color:"#00d4aa",type:"dashed"},label:{formatter:"主力线40",color:"#00d4aa",fontSize:10}}]}},
    ],
  },true);
}
function renderHoldings(trainH, testH){
  const render=(data,tbId,valId)=>{
    const tb=document.getElementById(tbId);
    if(!data||data.length===0){tb.innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">暂无</td></tr>';return}
    let total=0;
    tb.innerHTML=data.map(h=>{total+=h.value||0;return`
      <tr><td><b>${h.name}</b></td><td>${(h.shares/100).toFixed(0)}</td><td>${h.avg_cost}</td><td>${h.price}</td>
      <td class="${h.pnl_pct>0?'green':'red'}">${h.pnl_pct>0?'+':''}${h.pnl_pct}%</td></tr>`}).join('');
    document.getElementById(valId).textContent='(持仓市值 '+(total/10000).toFixed(0)+'万)';
  };
  render(trainH,'trainHoldingBody','trainHoldingVal');
  render(testH,'testHoldingBody','testHoldingVal');
}
function openConfig(){document.getElementById("configModal").classList.add("active")}
function closeConfig(){document.getElementById("configModal").classList.remove("active")}
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  涨停放量策略 · 可视化回测平台")
    print("=" * 55)
    print(f"  地址: http://127.0.0.1:5000")
    print(f"  股票池: {len(FULL_UNIVERSE)}只全行业")
    print(f"  架构: web_app.py → engine.py → indicators.py")
    print("=" * 55 + "\n")

    print("[预热] 加载数据...")
    for sym, name in FULL_UNIVERSE:
        key = f"stock_{sym}_2024-01-01_2025-12-31"
        if key not in _data_cache:
            df = _load_cache(sym, "2024-01-01", "2025-12-31")
            if df is not None: _data_cache[key] = df
    for idx_sym, idx_name in BENCHMARK_INDICES:
        key = f"idx_{idx_sym}_2024-01-01_2025-12-31"
        if key not in _data_cache:
            df = _load_cache(idx_sym, "2024-01-01", "2025-12-31")
            if df is not None: _data_cache[key] = df
    print("[预热] 数据就绪，运行回测...")
    _cached_result = run_dual_period()
    ts = _cached_result["train"]["summary"]
    vs = _cached_result["test"]["summary"]
    print(f"[预热] 训练集({TRAIN_PERIOD[0]}~{TRAIN_PERIOD[1]}): 收益={ts['total_return']}% 夏普={ts['sharpe_ratio']} 交易={ts['total_trades']}笔")
    print(f"[预热] 测试集({TEST_PERIOD[0]}~{TEST_PERIOD[1]}): 收益={vs['total_return']}% 夏普={vs['sharpe_ratio']} 交易={vs['total_trades']}笔")
    print()

    app.run(host="127.0.0.1", port=5000, debug=False)
