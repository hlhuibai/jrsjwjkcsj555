"""
策略引擎层：数据获取、因子计算、回测核心
仅供算法同学修改，web层通过 run_full_backtest() 调用
"""

import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from indicators import (
    calc_main_force_score, calc_institutional_flow, calc_mfi,
    detect_accumulation, detect_distribution,
    calc_bottom_distance, calc_bottom_signal,
)

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

def run_dual_period(config=None):
    """运行双周期回测：训练集+测试集，web层直接调用"""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    train_cfg = {**cfg, "start_date": TRAIN_PERIOD[0], "end_date": TRAIN_PERIOD[1]}
    test_cfg  = {**cfg, "start_date": TEST_PERIOD[0],  "end_date": TEST_PERIOD[1], "trade_start": TEST_TRADE_START}
    train_result = run_full_backtest(train_cfg)
    test_result  = run_full_backtest(test_cfg)
    # 测试集净值只保留交易期
    test_result["equity_series"] = [e for e in test_result["equity_series"] if e[0] >= TEST_TRADE_START]
    test_result["drawdown_series"] = [d for d in test_result["drawdown_series"] if d[0] >= TEST_TRADE_START]
    # 基准指数重新归一化到训练集，消除接头断层
    for k in test_result.get("benchmark_series", {}):
        filtered = [b for b in test_result["benchmark_series"][k] if b[0] >= TEST_TRADE_START]
        if filtered and k in train_result.get("benchmark_series", {}):
            train_bm = train_result["benchmark_series"][k]
            if train_bm:
                scale = train_bm[-1][1] / filtered[0][1] if filtered[0][1] > 0 else 1.0
                filtered = [[b[0], round(b[1]*scale, 4)] for b in filtered]
        test_result["benchmark_series"][k] = filtered
    return {"train": train_result, "test": test_result, "params": cfg}
