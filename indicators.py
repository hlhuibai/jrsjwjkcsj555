"""
A股主力资金追踪指标模块
追踪大主力动向：大单资金流、筹码集中度、异动量能、吸筹/派发识别
"""

import numpy as np
import pandas as pd


def rolling_sum(series, window):
    """滚动求和"""
    result = np.empty(len(series))
    result[:] = np.nan
    for i in range(window - 1, len(series)):
        result[i] = np.nansum(series[i - window + 1 : i + 1])
    return result


def rolling_mean(series, window):
    """滚动均值"""
    result = np.empty(len(series))
    result[:] = np.nan
    for i in range(window - 1, len(series)):
        result[i] = np.nanmean(series[i - window + 1 : i + 1])
    return result


def rolling_std(series, window):
    """滚动标准差"""
    result = np.empty(len(series))
    result[:] = np.nan
    for i in range(window - 1, len(series)):
        result[i] = np.nanstd(series[i - window + 1 : i + 1])
    return result


def ema(series, period):
    """指数加权移动平均"""
    result = np.empty(len(series))
    result[:] = np.nan
    k = 2.0 / (period + 1)
    for i in range(len(series)):
        if np.isnan(series[i]):
            continue
        if i == 0 or np.isnan(result[i - 1]):
            result[i] = series[i]
        else:
            result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result


# ─── 核心指标 ───────────────────────────────────────────────

def calc_mfi(high, low, close, volume, period=14):
    """
    资金流量指标 (Money Flow Index)
    MFI > 80: 超买，主力可能出货
    MFI < 20: 超卖，主力可能吸筹
    """
    n = len(close)
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    positive_flow = np.zeros(n)
    negative_flow = np.zeros(n)

    for i in range(1, n):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow[i] = raw_money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            negative_flow[i] = raw_money_flow[i]

    pos_sum = rolling_sum(positive_flow, period)
    neg_sum = rolling_sum(negative_flow, period)

    mfi = np.empty(n)
    mfi[:] = np.nan
    for i in range(period, n):
        ratio = pos_sum[i] / neg_sum[i] if neg_sum[i] != 0 else 1.0
        mfi[i] = 100.0 - (100.0 / (1.0 + ratio))

    return mfi


def calc_obv(close, volume):
    """能量潮 (On-Balance Volume) —— 量在价先"""
    n = len(close)
    obv = np.empty(n)
    obv[:] = np.nan
    obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def calc_vwap(high, low, close, volume):
    """成交量加权均价 —— 主力的成本锚点"""
    n = len(close)
    typical = (high + low + close) / 3.0
    cum_pv = np.empty(n)
    cum_vol = np.empty(n)
    cpv = 0.0
    cv = 0.0
    for i in range(n):
        cpv += typical[i] * volume[i]
        cv += volume[i]
        cum_pv[i] = cpv
        cum_vol[i] = cv
    vwap = cum_pv / np.where(cum_vol == 0, 1, cum_vol)
    return vwap


def calc_volume_ratio(volume, period=5, ma_period=20):
    """量比 —— 当日成交量 / N日均量"""
    vol_ma = rolling_mean(volume, ma_period)
    vr = volume / np.where(vol_ma == 0, 1, vol_ma)
    return vr


def calc_institutional_flow(high, low, close, volume, short_period=3, long_period=20):
    """
    主力资金流向估算（基于量价关系）
    核心逻辑：价涨量增 = 主力主动买入，价跌量增 = 主力主动卖出
    返回: 主力净流向值（正=流入，负=流出）
    """
    n = len(close)
    typical = (high + low + close) / 3.0
    price_change = np.empty(n)
    price_change[0] = 0
    for i in range(1, n):
        price_change[i] = typical[i] - typical[i - 1]

    # 分配成交量到买卖方向
    buy_vol = np.where(price_change > 0, volume, volume * 0.3)
    sell_vol = np.where(price_change < 0, volume, volume * 0.3)

    # 大单因子：波动越大、量越大，越可能是主力
    amplitude = (high - low) / np.where(close == 0, 1, close)
    weight = amplitude * volume / rolling_mean(volume, long_period)

    raw_flow = (buy_vol - sell_vol) * weight

    return ema(raw_flow, short_period)


def calc_chip_concentration(close, volume, period=60):
    """
    筹码集中度估算
    基于价格-成交量分布的离散程度
    值越低 = 筹码越集中 = 主力控盘度高
    """
    n = len(close)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(period, n):
        window_close = close[i - period : i + 1]
        window_vol = volume[i - period : i + 1]
        total_vol = np.nansum(window_vol)
        if total_vol == 0:
            result[i] = np.nan
            continue
        vwap_win = np.nansum(window_close * window_vol) / total_vol
        variance = np.nansum(window_vol * (window_close - vwap_win) ** 2) / total_vol
        result[i] = np.sqrt(variance) / np.abs(vwap_win) if vwap_win != 0 else np.nan

    return result


def calc_abnormal_volume(volume, base_period=20, threshold=1.5):
    """
    异常放量检测
    成交量超过均量 threshold 倍 = 有主力动作
    """
    vol_ma = rolling_mean(volume, base_period)
    ratio = volume / np.where(vol_ma == 0, 1, vol_ma)
    return (ratio > threshold).astype(float) * ratio


def calc_main_force_score(
    high, low, close, volume, amount=None, period=14
):
    """
    趋势强度评分 (0-100)
    专注于趋势行情判断：资金流向 + 均线排列 + 量能趋势 + 短期动量
    """
    n = len(close)
    scores = np.zeros(n)

    # 均线
    ma5 = rolling_mean(close, 5)
    ma10 = rolling_mean(close, 10)
    ma20 = rolling_mean(close, 20)

    # 资金流向
    inst_flow = calc_institutional_flow(high, low, close, volume)

    # 量能趋势
    vol_ma5 = rolling_mean(volume, 5)
    vol_ma20 = rolling_mean(volume, 20)

    for i in range(60, n):
        s = 0.0

        # ① 资金流向强度 (40分)
        flow_val = inst_flow[i]
        flow_std = np.nanstd(inst_flow[max(0,i-14):i+1])
        if flow_std > 0:
            s += 40.0 * min(max((flow_val / flow_std + 1.0) / 2.0, 0), 1)

        # ② 均线多头排列 (25分) — MA5 > MA10 > MA20
        if not any(np.isnan([ma5[i], ma10[i], ma20[i]])):
            if ma5[i] > ma10[i] > ma20[i]:
                s += 25.0  # 完美多头
            elif ma5[i] > ma20[i]:
                s += 15.0  # 部分多头

        # ③ 量能趋势 (20分) — 近期放量
        if not np.isnan(vol_ma5[i]) and not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0:
            vol_ratio = vol_ma5[i] / vol_ma20[i]
            if 1.0 <= vol_ratio <= 2.0:
                s += 20.0  # 温和放量
            elif vol_ratio > 2.0:
                s += 12.0  # 过度放量
            elif vol_ratio >= 0.8:
                s += 8.0   # 量能持平

        # ④ 短期动量 (15分) — 10日涨幅
        if i >= 10 and close[i-10] > 0:
            ret_10d = (close[i] - close[i-10]) / close[i-10]
            if 0.05 <= ret_10d <= 0.30:
                s += 15.0  # 温和上涨
            elif 0 <= ret_10d < 0.05:
                s += 8.0   # 横盘

        scores[i] = s

    return scores


def detect_accumulation(close, volume, window=20):
    """
    检测主力吸筹阶段
    特征：价格横盘或微跌 + 成交量温和放大 + 阳线多于阴线
    返回: 吸筹信号值
    """
    n = len(close)
    signal = np.zeros(n)

    for i in range(window, n):
        win_close = close[i - window : i + 1]
        win_vol = volume[i - window : i + 1]

        price_range = (np.max(win_close) - np.min(win_close)) / np.mean(win_close)
        up_days = np.sum(np.diff(win_close) > 0)

        vol_early = np.mean(win_vol[: window // 2])
        vol_late = np.mean(win_vol[window // 2 :])

        if price_range < 0.10 and up_days / window > 0.45 and vol_late > vol_early:
            signal[i] = 1.0

    return signal


def detect_distribution(close, volume, window=20):
    """
    检测主力派发阶段
    特征：高位放量滞涨 + 长上影线 + 阴线增多
    返回: 派发信号值
    """
    n = len(close)
    signal = np.zeros(n)

    for i in range(window, n):
        win_close = close[i - window : i + 1]
        win_vol = volume[i - window : i + 1]

        price_change = (win_close[-1] - win_close[0]) / win_close[0]
        vol_early = np.mean(win_vol[: window // 2])
        vol_late = np.mean(win_vol[window // 2 :])

        # 高位滞涨 + 放量
        roc60 = (close[i] - np.min(win_close)) / np.max(win_close)
        if roc60 > 0.7 and price_change < 0.03 and vol_late > vol_early * 1.3:
            signal[i] = 1.0

    return signal
