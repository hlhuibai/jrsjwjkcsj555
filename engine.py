"""
策略引擎层：数据获取、因子计算、回测核心
仅供算法同学修改，web层通过 run_full_backtest() 调用
"""

import os, sys, glob, datetime, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from indicators import (
    calc_main_force_score, calc_institutional_flow, calc_mfi,
    detect_accumulation, detect_distribution,
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

TRAIN_PERIOD = ("2021-06-01", "2023-12-31")
def _today_str():
    """返回最近交易日，周末自动回退到周五"""
    today = datetime.date.today()
    while today.weekday() >= 5:  # 周六/日 → 回退
        today = today - datetime.timedelta(days=1)
    return today.strftime("%Y-%m-%d")

TEST_PERIOD = ("2023-10-01", _today_str())   # 提前3月加载数据，给Q1提供lookback
TEST_TRADE_START = "2024-01-02"

_cached_result = None
_data_cache = {}
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data_cache")


# ═══════════════════════════════════════════════════════════
# 数据层（与之前相同）
# ═══════════════════════════════════════════════════════════

def _sym_to_tx(symbol):
    code, mkt = symbol.split(".")
    return f"{'sh' if mkt == 'SH' else 'sz'}{code}"

def _cache_path(symbol):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe = symbol.replace(".", "_").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.pkl")

def _load_cache(symbol):
    """按股票代码加载缓存，自动从旧格式迁移"""
    path = _cache_path(symbol)
    if os.path.exists(path):
        try: return pd.read_pickle(path)
        except: pass
    # 迁移旧格式 {symbol}_{start}_{end}.pkl → {symbol}.pkl
    safe = symbol.replace(".", "_").replace("/", "_")
    try:
        old_files = [f for f in os.listdir(_CACHE_DIR)
                     if f.startswith(safe + "_") and f.endswith(".pkl")]
        if old_files:
            best = max(old_files, key=lambda f: os.path.getsize(
                os.path.join(_CACHE_DIR, f)))
            df = pd.read_pickle(os.path.join(_CACHE_DIR, best))
            if df is not None:
                _save_cache(symbol, df)
                return df
    except: pass
    return None

def _save_cache(symbol, df):
    try: df.to_pickle(_cache_path(symbol))
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
        cols = ["close","symbol"]
        if "volume" in df.columns: cols.append("volume")
        if "amount" in df.columns: cols.append("amount")
        return df[cols]
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
    """获取个股数据：缓存增量更新 > akshare拉取 > 指数驱动合成"""
    if seed is None: seed = hash(symbol) % (2**31)
    cache_key = f"stock_{symbol}"
    t_start, t_end = pd.Timestamp(start), pd.Timestamp(end)

    # 1. 内存缓存命中 → 直接返回切片
    if cache_key in _data_cache:
        df = _data_cache[cache_key]
        if df is not None and len(df) > 0:
            d_min, d_max = df.index.min(), df.index.max()
            if d_min <= t_start and d_max >= t_end:
                result = df.loc[t_start:t_end]
                if len(result) >= 2: return result

    # 2. 磁盘缓存 → 增量拉取缺失部分
    df = _load_cache(symbol)
    if df is not None and len(df) > 0:
        d_min, d_max = df.index.min(), df.index.max()
        need_new = d_max < t_end
        need_old = d_min > t_start

        if not need_new and not need_old:
            _data_cache[cache_key] = df
            result = df.loc[t_start:t_end]
            if len(result) >= 2: return result

        if need_new:
            gap_days = (t_end - d_max).days
            if gap_days > 2:  # 差距≤2天通常是周末/假期，跳过无效HTTP
                fetch_start = (d_max + pd.Timedelta(days=1)).strftime("%Y%m%d")
                new_df = _fetch_tx_stock(symbol, fetch_start, end.replace("-", ""))
                if new_df is not None and not new_df.empty:
                    df = pd.concat([df, new_df])
                    df = df[~df.index.duplicated(keep="last")]
                    df.sort_index(inplace=True)

        if need_old:
            gap_days = (d_min - t_start).days
            if gap_days > 2:  # 差距≤2天（节假日错位），跳过
                fetch_end = (d_min - pd.Timedelta(days=1)).strftime("%Y%m%d")
                old_df = _fetch_tx_stock(symbol, start.replace("-", ""), fetch_end)
                if old_df is not None and not old_df.empty:
                    df = pd.concat([old_df, df])
                    df = df[~df.index.duplicated(keep="last")]
                    df.sort_index(inplace=True)

        _data_cache[cache_key] = df
        _save_cache(symbol, df)
        result = df.loc[t_start:t_end]
        if len(result) >= 2: return result

    # 3. 无缓存 → akshare全量拉取
    df = _fetch_tx_stock(symbol, start, end)
    if df is not None and not df.empty:
        _data_cache[cache_key] = df
        _save_cache(symbol, df)
        result = df.loc[t_start:t_end]
        if len(result) >= 2: return result

    # 4. 合成兜底
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
    cache_key = f"idx_{symbol}"
    t_start, t_end = pd.Timestamp(start), pd.Timestamp(end)

    # 1. 内存缓存
    if cache_key in _data_cache:
        df = _data_cache[cache_key]
        if df is not None and len(df) > 0:
            d_min, d_max = df.index.min(), df.index.max()
            if d_min <= t_start and d_max >= t_end:
                return df.loc[t_start:t_end]

    # 2. 磁盘缓存
    df = _load_cache(symbol)
    if df is not None and len(df) > 0:
        d_min, d_max = df.index.min(), df.index.max()
        if d_min <= t_start and d_max >= t_end:
            df["name"] = name
            _data_cache[cache_key] = df
            return df.loc[t_start:t_end]

    # 3. akshare拉取
    df = _fetch_ak_index(symbol, start, end)
    if df is not None and not df.empty:
        df["name"] = name
        _data_cache[cache_key] = df
        _save_cache(symbol, df)
        return df

    # 4. 合成兜底
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

def _load_all_mcap():
    """启动时一次性加载全部市值数据到内存"""
    try:
        mcap = {}
        for path in glob.glob(os.path.join(_CACHE_DIR, "mcap", "*.pkl")):
            code = os.path.basename(path).replace(".pkl", "")
            try:
                df = pd.read_pickle(path)
                if df is not None and len(df) > 0:
                    df["date"] = pd.to_datetime(df.iloc[:, 0])
                    df.set_index("date", inplace=True)
                    mcap[code] = df.iloc[:, 3] / 1e8  # 总市值列(索引3，单位元→亿)
            except: pass
        return mcap
    except: return {}

# 模块级缓存
_mcap_cache = _load_all_mcap()


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

    # 全市场成交额（上证+深证，BaoStock源），用于量能冰点判断
    _total_market_amt = None
    try:
        import baostock as bs
        bs.login()
        rs = bs.query_history_k_data_plus("sh.000001", "date,amount",
            start_date=cfg["start_date"], end_date=cfg["end_date"])
        sh = rs.get_data()
        rs2 = bs.query_history_k_data_plus("sz.399001", "date,amount",
            start_date=cfg["start_date"], end_date=cfg["end_date"])
        sz = rs2.get_data()
        bs.logout()
        if sh is not None and sz is not None and len(sh) > 0:
            sh["date"] = pd.to_datetime(sh["date"]); sh.set_index("date", inplace=True)
            sz["date"] = pd.to_datetime(sz["date"]); sz.set_index("date", inplace=True)
            _total_market_amt = (sh["amount"].astype(float) + sz["amount"].astype(float))
    except: pass

    cash = initial_capital
    positions = {}
    trades = []
    daily_values = []
    total_commission = 0.0
    lookback = 60
    max_analyze = 50
    _score_cache = {}

    def _is_one_word(df_sym, dt):
        """一字板检测：开盘=最高=最低=收盘，全天无波动"""
        if dt not in df_sym.index: return False, None
        row = df_sym.loc[dt]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        if h == l:  # 全天无价格波动 = 一字板
            return True, "涨停" if c >= o else "跌停"
        return False, None

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

        # Stage 1: 全市场扫涨停放量（不限底部，牛熊通吃）
        candidates = []
        for sym, info in all_data.items():
            df = info["df"]
            if date not in df.index: continue
            idx = df.index.get_loc(date)
            if idx < 20: continue
            # 5日内涨停(>=8%)且量异常放大 = 大资金进场信号
            # 双基线：20日均量(短线) + 60日均量(长线)，防止持续放量抬高基线导致漏检
            vol_ma20 = df["volume"].iloc[max(0,idx-20):idx+1].mean()
            vol_ma60 = df["volume"].iloc[max(0,idx-60):idx+1].mean()
            if vol_ma20 <= 0: continue
            breakout_score = 0.0
            for j in range(max(1, idx-10), idx+1):
                if j >= len(df): break
                pct = (df["close"].iloc[j] - df["close"].iloc[j-1]) / df["close"].iloc[j-1]
                vol = df["volume"].iloc[j]
                # 路径A: 短线暴量(>20日均量×2)  路径B: 持续放量但长基线仍异常
                vol_ok = vol > vol_ma20 * 2.0 or (vol > vol_ma20 * 1.2 and vol_ma60 > 0 and vol > vol_ma60 * 1.8)
                if pct >= 0.08 and vol_ok:
                    recency = 1.0 - (idx - j) / 10.0
                    vol_ratio = min(vol / max(vol_ma20, 1), 4.0)
                    breakout_score = 100.0 * recency * 0.5 + 100.0 * min(vol_ratio / 4.0, 1.0) * 0.5
                    break
            # 市值过滤：真实市值 > 300亿（内存查找，极快）
            code = sym.split(".")[0]
            if code in _mcap_cache:
                mcap_series = _mcap_cache[code]
                date_str = df.index[idx]
                # 找该日期之前最近的市值
                valid = mcap_series[mcap_series.index <= date_str]
                mcap_val = float(valid.iloc[-1]) if len(valid) > 0 else 0
            else:
                mcap_val = 999  # 无数据默认放行
            # 5日涨幅过滤：前5日已涨超30%不追（接盘风险）
            if idx >= 5:
                close_5d_ago = df["close"].iloc[idx-5]
                gain_5d = (df["close"].iloc[idx] - close_5d_ago) / close_5d_ago
                if gain_5d > 0.30: continue

            if breakout_score >= 50 and mcap_val > 100:
                candidates.append((sym, breakout_score, df, idx))

        # Stage 2: 涨停放量股做主力确认（TOP80）
        candidates.sort(key=lambda x: x[1], reverse=True)
        detailed = []
        for sym, bk_score, df, idx in candidates[:max_analyze]:
            ck = (sym, idx)
            if ck in _score_cache:
                mf = _score_cache[ck]
            else:
                hist = df.iloc[max(0,idx-120):idx+1]
                mf_arr = calc_main_force_score(hist["high"].values, hist["low"].values,
                                               hist["close"].values, hist["volume"].values, 14)
                mf = float(mf_arr[-1]) if not np.isnan(mf_arr[-1]) else 50.0
                _score_cache[ck] = mf
            detailed.append((sym, bk_score, mf))

        # 总资产
        tv = cash
        for s in list(positions.keys()):
            p = _p(s, date)
            if p is not None: tv += positions[s]["shares"] * p

        # ── 大盘风控：双MA分级 ──
        # Level 1 (risk_off):  沪深300<MA20 且缩量 → 仓位上限20%（牛回调）
        # Level 2 (bear_market): 沪深300<MA20 且<MA60 → 禁止开仓（熊市）
        risk_off = False
        bear_market = False
        hs300 = benchmark_data.get("000300.SH")
        if hs300 is not None and date in hs300.index:
            idx_hs = hs300.index.get_loc(date)
            if idx_hs >= 60:
                ma20 = float(hs300["close"].iloc[max(0,idx_hs-20):idx_hs+1].mean())
                ma60 = float(hs300["close"].iloc[max(0,idx_hs-60):idx_hs+1].mean())
                ma5_vol = float(hs300["volume"].iloc[max(0,idx_hs-5):idx_hs+1].mean())
                today_close = float(hs300["close"].iloc[idx_hs])
                today_vol = float(hs300["volume"].iloc[idx_hs])
                if today_close < ma20 and today_vol < ma5_vol:
                    if today_close < ma60:
                        bear_market = True
                    else:
                        risk_off = True
                # 全市场量能冰点 → 强制空仓（上证+深证成交额 < 8000亿）
                if _total_market_amt is not None and date in _total_market_amt.index:
                    if float(_total_market_amt.loc[date]) < 8000_0000_0000:
                        bear_market = True

        # 卖出：止损/半仓止盈/移动止损
        if date.strftime("%Y-%m-%d") >= trade_start:
            # 今日有买入信号的股票不卖出
            buy_candidates_today = {sym for sym, _, _ in detailed}
            for sym in list(positions.keys()):
                if sym in buy_candidates_today: continue  # 有买点则忽略卖点
                pos = positions[sym]
                if pos["shares"] <= 0: continue
                price = _p(sym, date)
                if price is None: continue

                # 一字跌停卖不出
                df_sym = all_data[sym]["df"]
                one_word, direction = _is_one_word(df_sym, date)
                if one_word and direction == "跌停": continue

                # 更新持仓最高价（用于移动止损）
                if price > pos.get("high_price", 0):
                    pos["high_price"] = price

                pnl = (price - pos["avg_cost"]) / pos["avg_cost"]
                dd_from_high = (price - pos["high_price"]) / pos["high_price"] if pos["high_price"] > 0 else 0
                half_taken = pos.get("half_taken", False)
                reason = None
                shares_to_sell = 0

                if half_taken:
                    # 已止盈半仓 → 剩余仓位移动止损15%
                    if dd_from_high <= -0.15:
                        reason = f"移动止损(回撤{dd_from_high*100:.1f}%)"
                        shares_to_sell = pos["shares"]
                else:
                    # 未止盈 → 止损8% 或 止盈20%减半仓
                    if pnl <= stop_loss:
                        reason = f"止损({pnl*100:.1f}%)"
                        shares_to_sell = pos["shares"]
                    elif pnl >= take_profit:
                        reason = f"止盈半仓(+{pnl*100:.1f}%)"
                        shares_to_sell = pos["shares"] // 200 * 100  # 卖一半，整手

                if reason and shares_to_sell >= 100:
                    proceeds = shares_to_sell * price
                    comm = max(proceeds * 0.00061, 5)
                    total_commission += comm
                    cash += proceeds - comm
                    pos["shares"] -= shares_to_sell
                    trades.append({"date":date.strftime("%Y-%m-%d"),"symbol":sym,
                        "name":all_data[sym]["name"],"action":"sell","price":round(price,2),
                        "shares":shares_to_sell,"amount":round(proceeds,0),
                        "pnl_pct":round(pnl*100,2),"reason":reason})
                    if pos["shares"] < 100:
                        del positions[sym]  # 不够一手，清掉
                    elif not half_taken:
                        pos["half_taken"] = True  # 标记已减半

        # ── 买入：当日以最高价×98.2%成交（模拟盘中追涨）──
        if date.strftime("%Y-%m-%d") >= trade_start:
            current_total_pct = (tv - cash) / tv if tv > 0 else 0
            max_allowed_pct = 0.20 if risk_off else 1.0

            if not bear_market:  # 熊市确认 → 跳过所有买入（卖出照常执行）
                for sym, bk_score, mf in detailed:
                    if len(positions) >= max_positions: break
                    if sym in positions: continue
                    if mf < 40: continue
                    if risk_off and current_total_pct >= max_allowed_pct: break

                    df_sym = all_data[sym]["df"]
                    if date not in df_sym.index: continue
                    # 一字涨停买不进
                    one_word, _ = _is_one_word(df_sym, date)
                    if one_word: continue
                    day_high = float(df_sym.loc[date, "high"])
                    price = day_high * 0.982
                    if price <= 0: continue

                    target_pct = 0.05 + (max_single_pct - 0.05) * (bk_score / 100.0)
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
                        "entry_date":date.strftime("%Y-%m-%d"),"high_price":price,
                        "half_taken":False}
                    trades.append({"date":date.strftime("%Y-%m-%d"),"symbol":sym,
                        "name":all_data[sym]["name"],"action":"buy","price":round(price,2),
                        "shares":shares,"amount":round(cost,0),"pnl_pct":0,
                        "reason":f"涨停放量 强度={bk_score:.0f} 仓={target_pct*100:.0f}%"})

        # 净值
        sv = sum(pos["shares"]*_p(s,date) for s,pos in positions.items() if _p(s,date) is not None)
        tv = cash + sv
        daily_values.append({"date":date.strftime("%Y-%m-%d"),"value":round(tv,2),
                             "cash":round(cash,2),"stock_value":round(sv,2)})

    # ── 最终持仓 ──
    holdings = []
    last_date = all_dates[-1]
    for sym, pos in positions.items():
        if pos["shares"] <= 0: continue
        p = _p(sym, last_date)
        if p is None: continue
        pnl = (p - pos["avg_cost"]) / pos["avg_cost"]
        holdings.append({
            "symbol": sym,
            "name": all_data[sym]["name"],
            "shares": pos["shares"],
            "avg_cost": round(pos["avg_cost"], 2),
            "price": round(p, 2),
            "pnl_pct": round(pnl * 100, 2),
            "value": round(pos["shares"] * p, 0),
        })

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
        "holdings": holdings,
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
    test_result["equity_series"] = [e for e in test_result["equity_series"] if e[0] > TEST_TRADE_START]
    test_result["drawdown_series"] = [d for d in test_result["drawdown_series"] if d[0] > TEST_TRADE_START]
    # 基准指数从数据起始日展示；归一化到训练集接头
    test_start = TEST_PERIOD[0]
    for k in test_result.get("benchmark_series", {}):
        filtered = [b for b in test_result["benchmark_series"][k] if b[0] > test_start]
        if filtered and k in train_result.get("benchmark_series", {}):
            train_bm = train_result["benchmark_series"][k]
            if train_bm:
                scale = train_bm[-1][1] / filtered[0][1] if filtered[0][1] > 0 else 1.0
                filtered = [[b[0], round(b[1]*scale, 4)] for b in filtered]
        test_result["benchmark_series"][k] = filtered
    return {"train": train_result, "test": test_result, "params": cfg}


# ═══════════════════════════════════════════════════════════
# 每日自动更新
# ═══════════════════════════════════════════════════════════

def update_daily_data():
    """每日增量拉取所有股票+指数的最新交易日数据，写入缓存。
    可在收盘后(15:30+)通过 Windows 任务计划 / cron 调用。"""
    today = _today_str()
    start = TRAIN_PERIOD[0]
    updated, failed = [], []

    for sym, name in FULL_UNIVERSE:
        try:
            df = generate_price_data(sym, start, today)
            if df is not None: updated.append(sym)
            else: failed.append(sym)
        except Exception as e:
            failed.append(f"{sym}: {e}")

    for idx_sym, idx_name in BENCHMARK_INDICES:
        try:
            df = generate_index_data(idx_sym, idx_name, start, today)
            if df is not None: updated.append(idx_sym)
            else: failed.append(idx_sym)
        except Exception as e:
            failed.append(f"{idx_sym}: {e}")

    return {"date": today, "updated": len(updated), "failed": len(failed),
            "failed_list": failed[:10]}


if __name__ == "__main__":
    import json
    result = update_daily_data()
    print(json.dumps(result, ensure_ascii=False, indent=2))
