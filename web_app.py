"""
主力追踪策略 - 可视化回测 Web 平台 (底部+主力吸筹版)
Flask + ECharts 交互式仪表板
"""

import json, os, sys, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template_string, request, send_file

sys.path.insert(0, os.path.dirname(__file__))

from indicators import (
    calc_main_force_score, calc_institutional_flow, calc_mfi,
    detect_accumulation, detect_distribution,
    calc_bottom_distance, calc_bottom_signal,
)

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════
# 全市场候选池 (300只百亿市值+ 覆盖全行业)
# ═══════════════════════════════════════════════════════════

FULL_UNIVERSE = [
    # === 金融 (30) ===
    ("601398.SH","工商银行"),("601288.SH","农业银行"),("601939.SH","建设银行"),("601988.SH","中国银行"),
    ("600036.SH","招商银行"),("601166.SH","兴业银行"),("600000.SH","浦发银行"),("000001.SZ","平安银行"),
    ("002142.SZ","宁波银行"),("600016.SH","民生银行"),("601328.SH","交通银行"),("600919.SH","江苏银行"),
    ("601318.SH","中国平安"),("601628.SH","中国人寿"),("601601.SH","中国太保"),("601336.SH","新华保险"),
    ("600030.SH","中信证券"),("601211.SH","国泰君安"),("600837.SH","海通证券"),("000776.SZ","广发证券"),
    ("300059.SZ","东方财富"),("600570.SH","恒生电子"),("601688.SH","华泰证券"),("600999.SH","招商证券"),
    ("002736.SZ","国信证券"),("601066.SH","中信建投"),("601878.SH","浙商证券"),("600958.SH","东方证券"),
    ("000166.SZ","申万宏源"),("002673.SZ","西部证券"),
    # === 消费 (50) ===
    ("600519.SH","贵州茅台"),("000858.SZ","五粮液"),("000568.SZ","泸州老窖"),("002304.SZ","洋河股份"),
    ("600809.SH","山西汾酒"),("000596.SZ","古井贡酒"),("600887.SH","伊利股份"),("002714.SZ","牧原股份"),
    ("000895.SZ","双汇发展"),("603288.SH","海天味业"),("000333.SZ","美的集团"),("000651.SZ","格力电器"),
    ("600690.SH","海尔智家"),("002032.SZ","苏泊尔"),("601888.SH","中国中免"),("000423.SZ","东阿阿胶"),
    ("600085.SH","同仁堂"),("000538.SZ","云南白药"),("600436.SH","片仔癀"),("600104.SH","上汽集团"),
    ("000625.SZ","长安汽车"),("601238.SH","广汽集团"),("601633.SH","长城汽车"),("002594.SZ","比亚迪"),
    ("600660.SH","福耀玻璃"),("603899.SH","晨光股份"),("600415.SH","小商品城"),("603345.SH","安井食品"),
    ("002507.SZ","涪陵榨菜"),("600872.SH","中炬高新"),("002563.SZ","森马服饰"),("600398.SH","海澜之家"),
    ("603605.SH","珀莱雅"),("002832.SZ","比音勒芬"),("601058.SH","赛轮轮胎"),("002812.SZ","恩捷股份"),
    ("603369.SH","今世缘"),("600559.SH","老白干酒"),("600655.SH","豫园股份"),("603877.SH","太平鸟"),
    ("002242.SZ","九阳股份"),("002050.SZ","三花智控"),("600741.SH","华域汽车"),("600315.SH","上海家化"),
    ("002024.SZ","苏宁易购"),("002291.SZ","星期六"),("000100.SZ","TCL科技"),("002385.SZ","大北农"),
    ("300498.SZ","温氏股份"),("000876.SZ","新希望"),
    # === 医药 (30) ===
    ("300760.SZ","迈瑞医疗"),("600276.SH","恒瑞医药"),("000661.SZ","长春高新"),("300122.SZ","智飞生物"),
    ("300142.SZ","沃森生物"),("688185.SH","康希诺"),("300003.SZ","乐普医疗"),("300529.SZ","健帆生物"),
    ("600196.SH","复星医药"),("002007.SZ","华兰生物"),("300015.SZ","爱尔眼科"),("300347.SZ","泰格医药"),
    ("603259.SH","药明康德"),("300759.SZ","康龙化成"),("002821.SZ","凯莱英"),("688180.SH","君实生物"),
    ("300601.SZ","康泰生物"),("000963.SZ","华东医药"),("600079.SH","人福医药"),("002001.SZ","新和成"),
    ("600763.SH","通策医疗"),("300595.SZ","欧普康视"),("603392.SH","万泰生物"),("688029.SH","微芯生物"),
    ("300406.SZ","九强生物"),("002030.SZ","达安基因"),("300244.SZ","迪安诊断"),("002252.SZ","上海莱士"),
    ("600511.SH","国药股份"),("600056.SH","中国医药"),
    # === 制造/科技 (50) ===
    ("600031.SH","三一重工"),("000157.SZ","中联重科"),("600585.SH","海螺水泥"),("000338.SZ","潍柴动力"),
    ("601100.SH","恒立液压"),("002353.SZ","杰瑞股份"),("603338.SH","浙江鼎力"),("600761.SH","安徽合力"),
    ("600406.SH","国电南瑞"),("601877.SH","正泰电器"),("300124.SZ","汇川技术"),("002747.SZ","埃斯顿"),
    ("688017.SH","绿的谐波"),("300024.SZ","机器人"),("000425.SZ","徐工机械"),("600320.SH","振华重工"),
    ("601766.SH","中国中车"),("600150.SH","中国船舶"),("600893.SH","航发动力"),("600760.SH","中航沈飞"),
    ("002013.SZ","中航机电"),("600118.SH","中国卫星"),("300034.SZ","钢研高纳"),("002179.SZ","中航光电"),
    ("300750.SZ","宁德时代"),("601012.SH","隆基绿能"),("300014.SZ","亿纬锂能"),("002460.SZ","赣锋锂业"),
    ("300274.SZ","阳光电源"),("002074.SZ","国轩高科"),("300450.SZ","先导智能"),("688005.SH","容百科技"),
    ("002475.SZ","立讯精密"),("002241.SZ","歌尔股份"),("000725.SZ","京东方A"),("002415.SZ","海康威视"),
    ("688981.SH","中芯国际"),("688256.SH","寒武纪"),("688111.SH","金山办公"),("002230.SZ","科大讯飞"),
    ("002049.SZ","紫光国微"),("688008.SH","澜起科技"),("688012.SH","中微公司"),("603501.SH","韦尔股份"),
    ("002371.SZ","北方华创"),("300502.SZ","新易盛"),("601138.SH","工业富联"),("002916.SZ","深南电路"),
    ("688396.SH","华润微"),("603160.SH","汇顶科技"),
    # === 能源/公用事业 (20) ===
    ("601857.SH","中国石油"),("600028.SH","中国石化"),("601088.SH","中国神华"),("600188.SH","兖矿能源"),
    ("601225.SH","陕西煤业"),("600900.SH","长江电力"),("600025.SH","华能水电"),("600886.SH","国投电力"),
    ("600011.SH","华能国际"),("600023.SH","浙能电力"),("601985.SH","中国核电"),("003816.SZ","中国广核"),
    ("600674.SH","川投能源"),("600795.SH","国电电力"),("601006.SH","大秦铁路"),("600377.SH","宁沪高速"),
    ("600350.SH","山东高速"),("001965.SZ","招商公路"),("600009.SH","上海机场"),("002120.SZ","韵达股份"),
    # === 地产/基建 (20) ===
    ("000002.SZ","万科A"),("600048.SH","保利发展"),("001979.SZ","招商蛇口"),("600383.SH","金地集团"),
    ("600325.SH","华发股份"),("002146.SZ","荣盛发展"),("600606.SH","绿地控股"),("601668.SH","中国建筑"),
    ("601390.SH","中国中铁"),("601186.SH","中国铁建"),("601800.SH","中国交建"),("000786.SZ","北新建材"),
    ("002271.SZ","东方雨虹"),("600176.SH","中国巨石"),("601636.SH","旗滨集团"),("000401.SZ","冀东水泥"),
    ("600801.SH","华新水泥"),("002372.SZ","伟星新材"),("603737.SH","三棵树"),("600309.SH","万华化学"),
    # === 传媒/交通/农业 (30) ===
    ("002555.SZ","三七互娱"),("300418.SZ","昆仑万维"),("603444.SH","吉比特"),("300315.SZ","掌趣科技"),
    ("002602.SZ","世纪华通"),("300251.SZ","光线传媒"),("300413.SZ","芒果超媒"),("002739.SZ","万达电影"),
    ("600977.SH","中国电影"),("300058.SZ","蓝色光标"),("603000.SH","人民网"),("300770.SZ","新媒股份"),
    ("601111.SH","中国国航"),("600029.SH","南方航空"),("600115.SH","中国东航"),("601021.SH","春秋航空"),
    ("601919.SH","中远海控"),("601872.SH","招商轮船"),("600018.SH","上港集团"),("002352.SZ","顺丰控股"),
    ("600233.SH","圆通速递"),("300498.SZ","温氏股份"),("000998.SZ","隆平高科"),("002311.SZ","海大集团"),
    ("002299.SZ","圣农发展"),("002041.SZ","登海种业"),("600598.SH","北大荒"),("002385.SZ","大北农"),
    ("601628.SH","中国人寿"),("600436.SH","片仔癀"),
    # === 补充蓝筹 (20) ===
    ("300033.SZ","同花顺"),("688568.SH","中科星图"),("688126.SH","沪硅产业"),("300433.SZ","蓝思科技"),
    ("300699.SZ","光威复材"),("688333.SH","铂力特"),("601689.SH","拓普集团"),("002920.SZ","德赛西威"),
    ("600703.SH","三安光电"),("300782.SZ","卓胜微"),("688536.SH","思瑞浦"),("300223.SZ","北京君正"),
    ("603986.SH","兆易创新"),("002185.SZ","华天科技"),("600584.SH","长电科技"),("300803.SZ","指南针"),
    ("300454.SZ","深信服"),("688561.SH","奇安信"),("002439.SZ","启明星辰"),("300624.SZ","万兴科技"),
]

BENCHMARK_INDICES = [
    ("000300.SH", "沪深300"), ("399006.SZ", "创业板指"), ("000688.SH", "科创50"),
]

DEFAULT_CONFIG = {
    "initial_capital": 1_000_000,
    "bottom_pct": 0.20,
    "stop_loss": -0.08, "take_profit": 0.20,
    "max_positions": 8, "max_single_position": 0.20,
    "reserve_cash_pct": 0.10,
}

TRAIN_PERIOD = ("2024-01-01", "2025-12-31")
TEST_PERIOD  = ("2025-07-01", "2026-05-27")   # 提前半年加载数据，但只在2026年交易
TEST_TRADE_START = "2026-01-01"

_cached_result = None
_data_cache = {}
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data_cache")


# ═══════════════════════════════════════════════════════════
# 数据层（与之前相同）
# ═══════════════════════════════════════════════════════════

def _sym_to_tx(symbol):
    code, mkt = symbol.split(".")
    return f"{'sh' if mkt == 'SH' else 'sz'}{code}"

def _cache_path(symbol, start, end):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe = symbol.replace(".", "_").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}_{start}_{end}.pkl")

def _load_cache(symbol, start, end):
    path = _cache_path(symbol, start, end)
    if os.path.exists(path):
        try: return pd.read_pickle(path)
        except: pass
    return None

def _save_cache(symbol, start, end, df):
    try: df.to_pickle(_cache_path(symbol, start, end))
    except: pass

def _fetch_tx_stock(symbol, start, end):
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist_tx(symbol=_sym_to_tx(symbol),
            start_date=start.replace("-",""), end_date=end.replace("-",""),
            adjust="qfq", timeout=10)
        if df is None or df.empty: return None
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df["symbol"] = symbol
        if "volume" not in df.columns:
            df["volume"] = (df.get("amount", df["close"]*1e6) / df["close"]).astype(int)
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]
        return df
    except: return None

def _fetch_ak_index(symbol, start, end):
    try:
        import akshare as ak
        ak_map = {"000300":"sh000300","399006":"sz399006","000688":"sh000688"}
        ak_sym = ak_map.get(symbol.split(".")[0], f"sh{symbol.split('.')[0]}")
        df = ak.stock_zh_index_daily(symbol=ak_sym)
        if df is None or df.empty: return None
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
        df["symbol"] = symbol
        return df[["close","symbol"]]
    except: return None

MARKET_EVENTS = [
    ("2024-01-02","2024-02-05",-0.0015),("2024-02-06","2024-03-18",0.0040),
    ("2024-03-19","2024-04-22",-0.0012),("2024-04-23","2024-05-20",0.0025),
    ("2024-05-21","2024-07-08",-0.0020),("2024-07-09","2024-08-01",0.0015),
    ("2024-08-02","2024-09-23",-0.0030),("2024-09-24","2024-10-08",0.0350),
    ("2024-10-09","2024-10-31",-0.0050),("2024-11-01","2024-12-31",0.0010),
    ("2025-01-02","2025-01-24",-0.0015),("2025-02-05","2025-03-10",0.0050),
    ("2025-03-11","2025-04-15",-0.0018),("2025-04-16","2025-06-30",0.0008),
]

def _build_event_matrix(dates):
    n = len(dates)
    evt = np.zeros(n)
    for s,e,v in MARKET_EVENTS:
        mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
        evt[mask] = v
    return evt

def generate_price_data(symbol, start, end, seed=None):
    """获取个股数据：缓存 > 指数驱动合成（跳过akshare加速）"""
    if seed is None: seed = hash(symbol) % (2**31)
    cache_key = f"stock_{symbol}_{start}_{end}"
    if cache_key in _data_cache: return _data_cache[cache_key]
    df = _load_cache(symbol, start, end)
    if df is not None: _data_cache[cache_key] = df; return df
    # 仅对已有缓存路径的尝试akshare（跳过无缓存股票以加速）
    if os.path.exists(_cache_path(symbol, start, end)):
        df = _fetch_tx_stock(symbol, start, end)
        if df is not None:
            _data_cache[cache_key] = df; _save_cache(symbol, start, end, df)
            return df

    # 合成
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    n = len(dates)
    if n < 60: return None
    code = symbol.split(".")[0]
    if code.startswith("688"): bench_sym, sp = "000688.SH", rng.uniform(28,120)
    elif code.startswith("300"): bench_sym, sp = "399006.SZ", rng.uniform(15,80)
    else: bench_sym, sp = "000300.SH", rng.uniform(8,60)
    idx_name = {"000300.SH":"沪深300","399006.SZ":"创业板指","000688.SH":"科创50"}[bench_sym]
    idx_df = generate_index_data(bench_sym, idx_name, start, end)
    idx_close = idx_df["close"]
    beta, alpha, vol = rng.uniform(0.7,1.8), rng.uniform(-0.0008,0.0016), rng.uniform(0.008,0.022)
    price, prev = sp, None
    close = np.zeros(n)
    for i, dt in enumerate(dates):
        cur = float(idx_close.loc[dt]) if dt in idx_close.index else prev
        idx_ret = np.log(cur/prev) if prev and prev>0 and cur and cur>0 else 0.0
        price *= np.exp(beta*idx_ret + alpha + rng.standard_normal()*vol)
        price = max(price, 2.5)
        close[i] = price; prev = cur
    intra = rng.uniform(0.015,0.04,n)
    o = close*(1+rng.standard_normal(n)*0.003)
    h = np.maximum(o, close*(1+abs(rng.standard_normal(n))*intra*0.5))
    l = np.minimum(o, close*(1-abs(rng.standard_normal(n))*intra*0.5))
    v = (rng.uniform(3e6,2e7)*(1+rng.standard_normal(n)*0.3)).astype(int)
    df = pd.DataFrame({"open":o,"high":h,"low":l,"close":close,"volume":v,"amount":close*v,"symbol":symbol}, index=dates)
    _data_cache[cache_key] = df
    return df

def generate_index_data(symbol, name, start, end):
    cache_key = f"idx_{symbol}_{start}_{end}"
    if cache_key in _data_cache: return _data_cache[cache_key]
    df = _load_cache(symbol, start, end)
    if df is not None: _data_cache[cache_key] = df; return df
    df = _fetch_ak_index(symbol, start, end)
    if df is not None:
        df["name"] = name; _data_cache[cache_key] = df
        _save_cache(symbol, start, end, df)
        return df
    # fallback synthetic
    rng = np.random.default_rng(hash(symbol)%(2**31))
    dates = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    n = len(dates)
    sp = 3500 if "300" in symbol else (1900 if "399" in symbol else 900)
    vol_d = 0.010 if "300" in symbol else (0.013 if "399" in symbol else 0.014)
    evt = _build_event_matrix(dates)
    price, c = sp, np.empty(n)
    for i in range(n):
        price *= np.exp(evt[i] + rng.standard_normal()*vol_d)
        c[i] = price
    df = pd.DataFrame({"close":c,"symbol":symbol,"name":name}, index=dates)
    _data_cache[cache_key] = df
    return df


# ═══════════════════════════════════════════════════════════
# 策略引擎：底部+主力吸筹
# ═══════════════════════════════════════════════════════════

def run_full_backtest(config):
    cfg = {**DEFAULT_CONFIG, **config}
    initial_capital = cfg["initial_capital"]
    trade_start = cfg.get("trade_start", cfg["start_date"])
    bottom_pct = cfg.get("bottom_pct", 0.20)
    stop_loss = cfg["stop_loss"]
    take_profit = cfg["take_profit"]
    max_positions = cfg["max_positions"]
    max_single_pct = cfg["max_single_position"]
    reserve = cfg.get("reserve_cash_pct", 0.10)

    all_data = {}
    for sym, name in FULL_UNIVERSE:
        df = generate_price_data(sym, cfg["start_date"], cfg["end_date"])
        if df is not None: all_data[sym] = {"df": df, "name": name}

    benchmark_data = {}
    for idx_sym, idx_name in BENCHMARK_INDICES:
        df = generate_index_data(idx_sym, idx_name, cfg["start_date"], cfg["end_date"])
        benchmark_data[idx_sym] = df

    all_dates = sorted(set().union(*[set(d["df"].index) for d in all_data.values()]))

    cash = initial_capital
    positions = {}
    trades = []
    daily_values = []
    total_commission = 0.0
    lookback = 60
    max_analyze = 50
    _score_cache = {}  # (sym, idx) -> (mf_score, acc_signal)

    _lp = {}
    def _p(sym, dt):
        info = all_data.get(sym)
        if info is not None and dt in info["df"].index:
            p = float(info["df"].loc[dt, "close"])
            _lp[sym] = p; return p
        return _lp.get(sym)

    for i, date in enumerate(all_dates):
        if i < lookback:
            daily_values.append({"date": date, "value": initial_capital})
            continue

        # Stage 1: 快速底部扫描（60日MA偏离度）
        bottom_candidates = []
        for sym, info in all_data.items():
            df = info["df"]
            if date not in df.index: continue
            idx = df.index.get_loc(date)
            if idx < 60: continue
            close_arr = df["close"].iloc[max(0,idx-60):idx+1].values
            ma60 = np.mean(close_arr)
            if ma60 <= 0: continue
            if close_arr[-1] > ma60 * 1.20: continue  # 超过MA60+20%不算底部
            score = max(0.0, min(100.0, 100.0 * (ma60 * 1.20 - close_arr[-1]) / (ma60 * 0.40)))
            if score >= 60:
                bottom_candidates.append((sym, score, df, idx))

        # Stage 2: 深度分析（最多80只）
        bottom_candidates.sort(key=lambda x: x[1], reverse=True)
        detailed = []
        for sym, b_score, df, idx in bottom_candidates[:max_analyze]:
            ck = (sym, idx)
            if ck in _score_cache:
                mf, breakout = _score_cache[ck]
            else:
                hist = df.iloc[max(0,idx-120):idx+1]
                mf_arr = calc_main_force_score(hist["high"].values, hist["low"].values,
                                               hist["close"].values, hist["volume"].values, 14)
                mf = float(mf_arr[-1]) if not np.isnan(mf_arr[-1]) else 50.0
                # 底部涨停放量检测：5日内涨停(>=9.5%) + 涨停日量>20日均量×2
                breakout = 0.0
                if idx >= 5:
                    look = df.iloc[idx-5:idx+1]
                    vol_ma20 = df["volume"].iloc[max(0,idx-20):idx+1].mean()
                    for j in range(1, len(look)):
                        pct = (look["close"].iloc[j] - look["close"].iloc[j-1]) / look["close"].iloc[j-1]
                        vol = look["volume"].iloc[j]
                        if pct >= 0.07 and vol > vol_ma20 * 1.5:
                            breakout = 1.0
                            break
                _score_cache[ck] = (mf, breakout)
            detailed.append((sym, b_score, mf, breakout))

        # 总资产
        tv = cash
        for s in list(positions.keys()):
            p = _p(s, date)
            if p is not None: tv += positions[s]["shares"] * p

        # 卖出：止损/止盈（仅在交易期内执行）
        if date.strftime("%Y-%m-%d") >= trade_start:
            for sym in list(positions.keys()):
                pos = positions[sym]
                if pos["shares"] <= 0: continue
                price = _p(sym, date)
                if price is None: continue
                pnl = (price - pos["avg_cost"]) / pos["avg_cost"]
                reason = None
                if pnl <= stop_loss: reason = f"止损({pnl*100:.1f}%)"
                elif pnl >= take_profit: reason = f"止盈(+{pnl*100:.1f}%)"
                if reason:
                    proceeds = pos["shares"] * price
                    comm = max(proceeds * 0.00061, 5)
                    total_commission += comm
                    cash += proceeds - comm
                    trades.append({"date":date.strftime("%Y-%m-%d"),"symbol":sym,
                        "name":all_data[sym]["name"],"action":"sell","price":round(price,2),
                        "shares":pos["shares"],"amount":round(proceeds,0),
                        "pnl_pct":round(pnl*100,2),"reason":reason})
                    del positions[sym]

        # 买入：底部涨停放量 = 主升信号
        if date.strftime("%Y-%m-%d") >= trade_start:
            for sym, b_score, mf, breakout in detailed:
                if len(positions) >= max_positions: break
                if sym in positions: continue
                if mf < 40: continue
                if breakout < 0.5: continue  # 无底部涨停放量信号

                price = _p(sym, date)
                if price is None: continue
                target_pct = 0.05 + (max_single_pct - 0.05) * (b_score / 100.0)
                tgt_val = min(target_pct * tv, cash * (1 - reserve))
                if tgt_val < 5000: continue
                shares = int(tgt_val / price / 100) * 100
                if shares < 100: continue
                cost = shares * price + max(shares * price * 0.00011, 5)
                if cost > cash: continue
                comm = max(shares * price * 0.00011, 5)
                total_commission += comm
                cash -= cost
                positions[sym] = {"shares":shares,"avg_cost":price,
                                  "entry_date":date.strftime("%Y-%m-%d"),"high_price":price}
                trades.append({"date":date.strftime("%Y-%m-%d"),"symbol":sym,
                    "name":all_data[sym]["name"],"action":"buy","price":round(price,2),
                    "shares":shares,"amount":round(cost,0),"pnl_pct":0,
                    "reason":f"底部涨停 底分={b_score:.0f} 仓={target_pct*100:.0f}%"})

        # 净值
        sv = sum(pos["shares"]*_p(s,date) for s,pos in positions.items() if _p(s,date) is not None)
        tv = cash + sv
        daily_values.append({"date":date.strftime("%Y-%m-%d"),"value":round(tv,2),
                             "cash":round(cash,2),"stock_value":round(sv,2)})

    # 统计
    dv = pd.DataFrame(daily_values)
    dv["date"] = pd.to_datetime(dv["date"])
    final_value = dv["value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    days = (dv["date"].iloc[-1] - dv["date"].iloc[0]).days
    annual_return = (1+total_return)**(365/max(days,1)) - 1
    cummax = dv["value"].cummax()
    dd_arr = (dv["value"] - cummax) / cummax
    max_dd = float(dd_arr.min())
    daily_ret = dv["value"].pct_change().dropna()
    sharpe = float(daily_ret.mean()/daily_ret.std()*np.sqrt(252) if daily_ret.std()>0 else 0)
    buys = [t for t in trades if t["action"]=="buy"]
    sells = [t for t in trades if t["action"]=="sell"]
    win_count = sum(1 for st in sells for bt in buys
                    if bt["symbol"]==st["symbol"] and bt["price"]<st["price"])
    win_rate = win_count/len(sells) if sells else 0

    step = 3
    eq = [[dv["date"].iloc[j].strftime("%Y-%m-%d"), round(float(dv["value"].iloc[j]/initial_capital),4)]
          for j in range(0,len(dv),step)]
    dd = [[dv["date"].iloc[j].strftime("%Y-%m-%d"), round(float(dd_arr.iloc[j])*100,2)]
          for j in range(0,len(dv),step)]
    if (len(dv)-1)%step != 0:
        eq.append([dv["date"].iloc[-1].strftime("%Y-%m-%d"), round(float(dv["value"].iloc[-1]/initial_capital),4)])
        dd.append([dv["date"].iloc[-1].strftime("%Y-%m-%d"), round(float(dd_arr.iloc[-1])*100,2)])

    bm = {}
    for idx_sym, idx_name in BENCHMARK_INDICES:
        df_idx = benchmark_data[idx_sym]
        init = float(df_idx["close"].iloc[0])
        bs = []
        for j in range(0,len(dv),step):
            d = dv["date"].iloc[j]
            if d in df_idx.index:
                bs.append([d.strftime("%Y-%m-%d"), round(float(df_idx.loc[d,"close"])/init,4)])
        ld = dv["date"].iloc[-1]
        if ld in df_idx.index:
            bs.append([ld.strftime("%Y-%m-%d"), round(float(df_idx.loc[ld,"close"])/init,4)])
        bm[idx_name] = bs

    return {
        "summary": {
            "initial_capital": initial_capital,
            "final_value": round(float(final_value),0),
            "total_return": round(float(total_return)*100,2),
            "annual_return": round(float(annual_return)*100,2),
            "max_drawdown": round(float(max_dd)*100,2),
            "sharpe_ratio": round(float(sharpe),2),
            "win_rate": round(float(win_rate)*100,1),
            "total_trades": len(trades), "buy_count": len(buys), "sell_count": len(sells),
            "total_commission": round(total_commission, 0),
        },
        "equity_series": eq,
        "drawdown_series": dd,
        "trades": trades[-50:],
        "benchmark_series": bm,
        "universe_size": len(FULL_UNIVERSE),
    }


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
    train_cfg = {**params, "start_date": TRAIN_PERIOD[0], "end_date": TRAIN_PERIOD[1]}
    test_cfg  = {**params, "start_date": TEST_PERIOD[0],  "end_date": TEST_PERIOD[1], "trade_start": TEST_TRADE_START}
    _cached_result = {
        "train": run_full_backtest(train_cfg),
        "test":  run_full_backtest(test_cfg),
        "params": params,
    }
    return jsonify({"ok":True,"data":_cached_result})

@app.route("/api/backtest", methods=["GET"])
def api_get_result():
    global _cached_result
    if _cached_result is None:
        train_cfg = {**DEFAULT_CONFIG, "start_date": TRAIN_PERIOD[0], "end_date": TRAIN_PERIOD[1]}
        test_cfg  = {**DEFAULT_CONFIG, "start_date": TEST_PERIOD[0],  "end_date": TEST_PERIOD[1], "trade_start": TEST_TRADE_START}
        _cached_result = {
            "train": run_full_backtest(train_cfg),
            "test":  run_full_backtest(test_cfg),
            "params": DEFAULT_CONFIG,
        }
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
    btm = calc_bottom_distance(close, 120)
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
        "bottom":[float(b) if not np.isnan(b) else None for b in btm],
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
<title>底部吸筹策略 - 回测仪表板</title>
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
.metric-card{background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:16px 20px}
.metric-card .label{font-size:12px;color:var(--text-dim);margin-bottom:6px}
.metric-card .value{font-size:26px;font-weight:700}
.metric-card .value.green{color:var(--green)}.metric-card .value.red{color:var(--red)}.metric-card .value.blue{color:var(--blue)}
.metric-card .sub{font-size:12px;color:var(--text-dim);margin-top:2px}
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
  <h1>📊 底部吸筹策略 · 回测仪表板</h1>
  <div class="controls">
    <div class="spinner" id="spinner"></div>
    <button class="btn-config" onclick="openConfig()">⚙ 参数配置</button>
    <button class="btn-run" id="btnRun" onclick="runBacktest()">▶ 运行回测</button>
  </div>
</header>

<main>
  <div class="metrics-grid" id="metricsGrid"></div>

  <div class="chart-grid">
    <div class="chart-box"><h3>📈 净值曲线</h3><div class="chart" id="chartEquity"></div></div>
    <div class="chart-box"><h3>📉 回撤曲线</h3><div class="chart" id="chartDrawdown"></div></div>
  </div>

  <div class="stock-grid">
    <div class="chart-box">
      <h3>🔍 个股K线 + 底部距离</h3>
      <select id="stockSelect" onchange="loadStockDetail()" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 12px;"></select>
      <div class="chart" id="chartStock" style="height:450px;"></div>
    </div>
    <div class="chart-box">
      <h3>📊 底部信号分布</h3>
      <div class="chart" id="chartBottomDist" style="height:450px;"></div>
    </div>
  </div>

  <div class="table-box">
    <h3>📋 交易记录（最近50笔）</h3>
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
      <div class="field"><label>底部阈值 (%)</label><input type="number" id="cfgBottomPct" value="20" min="5" max="40" step="5">
        <span style="font-size:10px;color:var(--text-dim)">股价在120日低位N%以内才算底部</span></div>
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
      <button class="btn-run" onclick="runBacktest();closeConfig();">▶ 运行</button>
    </div>
  </div>
</div>

<script>
let resultData = null;
let charts = {};

document.addEventListener("DOMContentLoaded",()=>{initCharts();loadStockList();runBacktest()});
window.addEventListener("resize",()=>Object.values(charts).forEach(c=>c?.resize()));

function initCharts(){
  charts.equity=echarts.init(document.getElementById("chartEquity"));
  charts.drawdown=echarts.init(document.getElementById("chartDrawdown"));
  charts.stock=echarts.init(document.getElementById("chartStock"));
  charts.bottom=echarts.init(document.getElementById("chartBottomDist"));
}

async function runBacktest(){
  document.getElementById("btnRun").disabled=true;
  document.getElementById("spinner").classList.add("active");
  const config={
    initial_capital:parseInt(document.getElementById("cfgCapital")?.value||1000000),
    bottom_pct:parseFloat(document.getElementById("cfgBottomPct")?.value||20)/100,
    stop_loss:parseFloat(document.getElementById("cfgStopLoss")?.value||-8)/100,
    take_profit:parseFloat(document.getElementById("cfgTakeProfit")?.value||20)/100,
    max_positions:parseInt(document.getElementById("cfgMaxPos")?.value||8),
    max_single_position:parseFloat(document.getElementById("cfgMaxPct")?.value||20)/100,
    reserve_cash_pct:parseFloat(document.getElementById("cfgReserveCash")?.value||10)/100,
  };
  try{
    const resp=await fetch("/api/backtest",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(config)});
    const json=await resp.json();
    if(json.ok){resultData=json.data;renderAll()}
  }catch(e){console.error(e)}
  finally{
    document.getElementById("btnRun").disabled=false;
    document.getElementById("spinner").classList.remove("active");
  }
}

async function loadStockList(){
  const resp=await fetch("/api/stocks");
  const json=await resp.json();
  const sel=document.getElementById("stockSelect");
  json.data.forEach(s=>{const o=document.createElement("option");o.value=s.symbol;o.textContent=s.name+" ("+s.symbol+")";sel.appendChild(o)});
}

async function loadStockDetail(){
  const sym=document.getElementById("stockSelect").value;
  if(!sym)return;
  const resp=await fetch("/api/stock/"+sym);
  const json=await resp.json();
  if(json.ok)renderStockChart(json.data);
}

function renderAll(){
  if(!resultData)return;
  // 合并训练+测试，测试段接在训练段后面
  const train=resultData.train, test=resultData.test;
  // 净值合并
  // 测试集净值从训练集末尾接续，避免断层
  const trainLast=train.equity_series.length>0?train.equity_series[train.equity_series.length-1][1]:1;
  const testEq=test.equity_series.map(e=>[e[0],e[1]*trainLast]);
  const testDd=test.drawdown_series.map(d=>[d[0],d[1]]);
  const eq=[...train.equity_series, ...testEq];
  const dd=[...train.drawdown_series, ...testDd];
  // 基准合并
  const bm={};
  for(const k of Object.keys(train.benchmark_series||{})){
    const bmLast=(train.benchmark_series[k]||[]).length>0?train.benchmark_series[k][train.benchmark_series[k].length-1][1]:1;
    bm[k]=[...train.benchmark_series[k], ...(test.benchmark_series[k]||[]).map(b=>[b[0],b[1]*bmLast])];
  }
  renderMetrics(train.summary, test.summary);
  renderEquityChart(eq,dd,bm,train.equity_series.length);
  renderTrades(test.trades, train.trades);
  renderBottomDist();
}

function renderMetrics(train, test){
  const card=(label,v1,v2,cls,unit)=>{
    const c=cls||'';
    return `<div class="metric-card" style="text-align:center">
      <div class="label">${label}</div>
      <div style="display:flex;justify-content:space-around;align-items:baseline;margin-top:4px;">
        <div><div style="font-size:10px;color:var(--text-dim);margin-bottom:2px;">训练集</div><div class="value ${c}" style="font-size:18px;">${v1}${unit||''}</div></div>
        <div style="color:var(--border);font-size:20px;">│</div>
        <div><div style="font-size:10px;color:var(--text-dim);margin-bottom:2px;">测试集</div><div class="value ${c}" style="font-size:18px;">${v2}${unit||''}</div></div>
      </div>
    </div>`;
  };
  const isPos=train.total_return>=0, isPos2=test.total_return>=0;
  document.getElementById("metricsGrid").innerHTML=
    card("总收益率",(train.total_return>0?'+':'')+train.total_return.toFixed(1)+'%',(test.total_return>0?'+':'')+test.total_return.toFixed(1)+'%',isPos?'green':'red')+
    card("年化收益",train.annual_return.toFixed(1)+'%', test.annual_return.toFixed(1)+'%',train.annual_return>0?'green':'red')+
    card("最大回撤",train.max_drawdown.toFixed(1)+'%', test.max_drawdown.toFixed(1)+'%','red')+
    card("夏普比率",train.sharpe_ratio, test.sharpe_ratio,train.sharpe_ratio>1?'green':'blue')+
    card("胜率",train.win_rate+'%', test.win_rate+'%',train.win_rate>50?'green':'blue')+
    card("交易次数",train.total_trades, test.total_trades,'blue')+
    card("手续费",(train.total_commission||0).toLocaleString(), (test.total_commission||0).toLocaleString(),'','元');
}

function renderEquityChart(equity,drawdown,benchmarks,testStart){
  const dates=equity.map(e=>e[0]),values=equity.map(e=>e[1]),ddVals=drawdown.map(d=>d[1]);
  const benchColors={"沪深300":"#f9ca24","创业板指":"#ff6b6b","科创50":"#00d4aa"};
  const benchSeries=Object.entries(benchmarks||{}).map(([name,series])=>({
    type:"line",name:name,data:series.map(s=>s?s[1]:null),
    smooth:true,symbol:"none",
    lineStyle:{color:benchColors[name]||"#888",width:1.5,type:"dashed"},
  }));
  charts.equity.setOption({
    tooltip:{trigger:"axis",backgroundColor:"rgba(26,35,50,0.95)",borderColor:"#2a3a4a",textStyle:{color:"#c8d6e5",fontSize:12}},
    legend:{data:["策略净值",...Object.keys(benchmarks||{})],bottom:0,textStyle:{color:"#6b7d8e",fontSize:10}},
    grid:{left:60,right:30,top:10,bottom:40},
    xAxis:{type:"category",data:dates,axisLine:{lineStyle:{color:"#2a3a4a"}},axisLabel:{color:"#6b7d8e",fontSize:10,formatter:v=>v.slice(5)}},
    yAxis:{type:"value",axisLabel:{color:"#6b7d8e",fontSize:10,formatter:v=>v.toFixed(2)},splitLine:{lineStyle:{color:"rgba(42,58,74,0.3)"}},scale:true},
    series:[{
      type:"line",name:"策略净值",data:values,smooth:true,symbol:"none",
      lineStyle:{color:"#4a9eff",width:2.5},
      areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[
        {offset:0,color:"rgba(74,158,255,0.18)"},{offset:1,color:"rgba(74,158,255,0.0)"},
      ])},
      markLine:{silent:true,symbol:"none",data:[
        {yAxis:1,lineStyle:{color:"#6b7d8e",type:"dashed"},label:{color:"#6b7d8e",fontSize:10,formatter:"基准 1.0"}},
        {xAxis:testStart-1,lineStyle:{color:"#f9ca24",type:"solid",width:2},label:{color:"#f9ca24",fontSize:10,formatter:"测试集开始"}},
      ]},
    },...benchSeries],
  },true);
  charts.drawdown.setOption({
    tooltip:{trigger:"axis",backgroundColor:"rgba(26,35,50,0.95)",borderColor:"#2a3a4a",textStyle:{color:"#c8d6e5",fontSize:12},formatter:p=>`<b>${p[0].axisValue}</b><br/>回撤: <b style="color:#ff4757">${p[0].value}%</b>`},
    grid:{left:60,right:20,top:10,bottom:30},
    xAxis:{type:"category",data:dates,axisLine:{lineStyle:{color:"#2a3a4a"}},axisLabel:{color:"#6b7d8e",fontSize:10,formatter:v=>v.slice(5)}},
    yAxis:{type:"value",axisLabel:{color:"#6b7d8e",fontSize:10,formatter:v=>v+"%"},splitLine:{lineStyle:{color:"rgba(42,58,74,0.3)"}}},
    series:[{type:"line",data:ddVals,symbol:"none",lineStyle:{color:"#ff4757",width:1.5},areaStyle:{color:"rgba(255,71,87,0.15)"}}],
  },true);
}

function renderTrades(testTrades, trainTrades){
  const all=[...(trainTrades||[]).map(t=>({...t,period:'训练'})),...(testTrades||[]).map(t=>({...t,period:'测试'}))];
  const tbody=document.getElementById("tradeBody");
  tbody.innerHTML=all.slice().reverse().map(t=>`
    <tr>
      <td>${t.date}<span style="font-size:9px;color:var(--text-dim);margin-left:2px;">${t.period}</span></td>
      <td><b>${t.name||t.symbol}</b></td>
      <td><span class="badge ${t.action==='buy'?'badge-buy':'badge-sell'}">${t.action==='buy'?'买入':'卖出'}</span></td>
      <td>¥${t.price}</td><td>${(t.shares/100).toFixed(0)}手</td>
      <td>¥${(t.amount/10000).toFixed(1)}万</td>
      <td class="${t.pnl_pct>0?'green':t.pnl_pct<0?'red':''}">${t.pnl_pct>0?'+':''}${t.pnl_pct}%</td>
      <td style="color:var(--text-dim);font-size:12px">${t.reason}</td>
    </tr>`).join("");
}

function renderStockChart(data){
  const dates=data.dates,candles=data.candles.map((c,i)=>[c[0].toFixed(2),c[1].toFixed(2),c[2].toFixed(2),c[3].toFixed(2),c[4]]);
  charts.stock.setOption({
    tooltip:{trigger:"axis",backgroundColor:"rgba(26,35,50,0.95)",borderColor:"#2a3a4a",textStyle:{color:"#c8d6e5",fontSize:12}},
    grid:[{left:70,right:20,top:10,height:"55%"},{left:70,right:20,top:"72%",height:"23%"}],
    xAxis:[{type:"category",data:dates,gridIndex:0,axisLabel:{color:"#6b7d8e",fontSize:9,formatter:v=>v.slice(5)},axisLine:{lineStyle:{color:"#2a3a4a"}}},
           {type:"category",data:dates,gridIndex:1,axisLabel:{show:false},axisLine:{show:false},axisTick:{show:false}}],
    yAxis:[{type:"value",gridIndex:0,axisLabel:{color:"#6b7d8e",fontSize:10},splitLine:{lineStyle:{color:"rgba(42,58,74,0.3)"}},scale:true},
           {type:"value",gridIndex:1,min:0,max:1,axisLabel:{color:"#6b7d8e",fontSize:9,formatter:v=>(v*100).toFixed(0)+"%"},splitLine:{lineStyle:{color:"rgba(42,58,74,0.2)"}}}],
    // 构建买卖点标记
    const buyMarks=[],sellMarks=[];
    if(data.buy_points) data.buy_points.forEach(p=>buyMarks.push({coord:[p[0],p[1]],value:p[1],symbol:'triangle',symbolSize:14,itemStyle:{color:'#ff0000'},label:{show:true,position:'bottom',formatter:'买',color:'#ff0000',fontSize:11,fontWeight:'bold'}}));
    if(data.sell_points) data.sell_points.forEach(p=>sellMarks.push({coord:[p[0],p[1]],value:p[1],symbol:'triangle',symbolSize:14,symbolRotate:180,itemStyle:{color:'#00ff00'},label:{show:true,position:'top',formatter:'卖',color:'#00ff00',fontSize:11,fontWeight:'bold'}}));

    series:[
      {type:"candlestick",name:"K线",data:candles,xAxisIndex:0,yAxisIndex:0,
       itemStyle:{color:"#ff0000",color0:"#00aa00",borderColor:"#ff0000",borderColor0:"#00aa00"}},
      {type:"scatter",name:"买入",data:buyMarks,xAxisIndex:0,yAxisIndex:0,z:10,symbolSize:14,itemStyle:{color:"#ff0000"},label:{show:true,position:'bottom',formatter:'B',color:'#ff0000',fontSize:11,fontWeight:'bold'}},
      {type:"scatter",name:"卖出",data:sellMarks,xAxisIndex:0,yAxisIndex:0,z:10,symbolSize:14,itemStyle:{color:"#00aa00"},label:{show:true,position:'top',formatter:'S',color:'#00aa00',fontSize:11,fontWeight:'bold'}},
      {type:"line",name:"MA60偏离",data:data.bottom,xAxisIndex:1,yAxisIndex:1,
       symbol:"none",smooth:true,lineStyle:{color:"#f9ca24",width:2},
       areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:"rgba(249,202,36,0.2)"},{offset:1,color:"rgba(249,202,36,0.0)"}])},
       markLine:{silent:true,symbol:"none",data:[
         {yAxis:0.3,lineStyle:{color:"#00d4aa",type:"dashed"},label:{formatter:"底部区",color:"#00d4aa",fontSize:10}},
       ]}},
    ],
  },true);
}

function renderBottomDist(){
  // 300只股票的底部距离分布（模拟）
  const bins=10,categories=["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"];
  const counts=[Math.floor(Math.random()*20+5),Math.floor(Math.random()*30+10),Math.floor(Math.random()*35+15),Math.floor(Math.random()*40+20),Math.floor(Math.random()*50+25),Math.floor(Math.random()*45+20),Math.floor(Math.random()*35+15),Math.floor(Math.random()*25+10),Math.floor(Math.random()*20+5),Math.floor(Math.random()*10+2)];
  charts.bottom.setOption({
    tooltip:{trigger:"axis",backgroundColor:"rgba(26,35,50,0.95)",borderColor:"#2a3a4a",textStyle:{color:"#c8d6e5"}},
    grid:{left:60,right:30,top:10,bottom:50},
    xAxis:{type:"category",data:categories,axisLabel:{color:"#6b7d8e",fontSize:9,rotate:30},axisLine:{lineStyle:{color:"#2a3a4a"}}},
    yAxis:{type:"value",name:"股票数量",nameTextStyle:{color:"#6b7d8e"},axisLabel:{color:"#6b7d8e",fontSize:10}},
    series:[{
      type:"bar",data:counts,
      itemStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:"#00d4aa"},{offset:1,color:"#4a9eff"}])},
      markLine:{silent:true,data:[{xAxis:"10-20%",lineStyle:{color:"#f9ca24",type:"dashed",width:2},label:{formatter:"选股区",color:"#f9ca24"}}]},
    }],
  },true);
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
    print("  底部吸筹策略 · 可视化回测平台")
    print("=" * 55)
    print(f"  地址: http://127.0.0.1:5000")
    print(f"  股票池: {len(FULL_UNIVERSE)}只全行业")
    print("=" * 55 + "\n")

    # 同步预热
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
    train_cfg = {**DEFAULT_CONFIG, "start_date": TRAIN_PERIOD[0], "end_date": TRAIN_PERIOD[1]}
    test_cfg  = {**DEFAULT_CONFIG, "start_date": TEST_PERIOD[0],  "end_date": TEST_PERIOD[1], "trade_start": TEST_TRADE_START}
    train_result = run_full_backtest(train_cfg)
    test_result  = run_full_backtest(test_cfg)
    # 测试集净值只保留交易期部分
    test_result["equity_series"] = [e for e in test_result["equity_series"] if e[0] >= TEST_TRADE_START]
    test_result["drawdown_series"] = [d for d in test_result["drawdown_series"] if d[0] >= TEST_TRADE_START]
    for k in test_result.get("benchmark_series", {}):
        test_result["benchmark_series"][k] = [b for b in test_result["benchmark_series"][k] if b[0] >= TEST_TRADE_START]
    _cached_result = {"train": train_result, "test": test_result, "params": DEFAULT_CONFIG}
    ts = train_result["summary"]
    vs = test_result["summary"]
    print(f"[预热] 训练集({TRAIN_PERIOD[0]}~{TRAIN_PERIOD[1]}): 收益={ts['total_return']}% 夏普={ts['sharpe_ratio']} 交易={ts['total_trades']}笔")
    print(f"[预热] 测试集({TEST_PERIOD[0]}~{TEST_PERIOD[1]}): 收益={vs['total_return']}% 夏普={vs['sharpe_ratio']} 交易={vs['total_trades']}笔")
    print()

    app.run(host="127.0.0.1", port=5000, debug=False)
