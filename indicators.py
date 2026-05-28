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
    主力综合评分 (0-100)
    整合多个维度的信号生成综合评分：
    - 资金流向
    - 量价关系
    - 筹码集中度
    - 异常放量
    """
    n = len(close)

    # 子指标
    institutional_flow = calc_institutional_flow(high, low, close, volume)
    mfi = calc_mfi(high, low, close, volume, period)
    vr = calc_volume_ratio(volume, ma_period=20)
    concentration = calc_chip_concentration(close, volume, 60)
    abnormal_vol = calc_abnormal_volume(volume, 20, 1.5)

    # 统一量纲到 0-100
    scores = np.zeros(n)

    for i in range(max(period, 60), n):
        s = 0.0

        # 1. 机构资金流强度 (权重 35%)
        flow_val = institutional_flow[i]
        flow_std = np.nanstd(institutional_flow[i - period : i + 1])
        if flow_std > 0:
            s += 35.0 * min(max((flow_val / flow_std + 1.0) / 2.0, 0), 1)

        # 2. MFI 位置 (权重 20%)
        mfi_val = mfi[i]
        if not np.isnan(mfi_val):
            if 30 <= mfi_val <= 70:
                s += 20.0  # 健康区间
            elif mfi_val < 30:
                s += 20.0 * (1.0 - (30 - mfi_val) / 30)  # 超卖，可能吸筹
            else:
                s += 20.0 * max(1.0 - (mfi_val - 70) / 30, 0)  # 超买降温

        # 3. 量比 (权重 20%)
        vr_val = vr[i]
        if not np.isnan(vr_val):
            if 1.0 <= vr_val <= 2.5:
                s += 20.0  # 温和放量，主力稳步推进
            elif 0.6 <= vr_val < 1.0:
                s += 12.0  # 缩量，观望
            elif vr_val > 2.5:
                s += 10.0  # 过度放量，警惕出货

        # 4. 筹码集中度 (权重 15%)
        conc_val = concentration[i]
        if not np.isnan(conc_val):
            # 集中度越低越好（0.05 以下 = 高度集中）
            if conc_val < 0.05:
                s += 15.0
            elif conc_val < 0.10:
                s += 10.0
            elif conc_val < 0.15:
                s += 5.0

        # 5. 异常放量信号 (权重 10%)
        ab_val = abnormal_vol[i]
        if ab_val > 0:
            s += min(ab_val / 4.0, 1.0) * 10.0

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


def calc_bottom_distance(close, window=60):
    """
    股价相对60日均线的偏离度 (0-1)
    0 = 价格在MA60上方（非底部）
    1 = 价格远低于MA60（深度底部）
    """
    n = len(close)
    result = np.full(n, np.nan)
    ma = rolling_mean(close, window)

    for i in range(window, n):
        if np.isnan(ma[i]) or ma[i] <= 0:
            continue
        # 偏离度：低于MA60越多值越大
        deviation = (ma[i] - close[i]) / ma[i]
        result[i] = max(0.0, min(1.0, deviation / 0.30))  # 偏离30%封顶

    return result


def calc_bottom_signal(close, volume, window=60, bottom_pct=0.20):
    """
    底部+吸筹综合信号 (基于60日均线)
    底部条件：股价不超过MA60上方20%（即 close <= MA60 * 1.20）
    """
    n = len(close)
    ma = rolling_mean(close, window)
    acc = detect_accumulation(close, volume, 20)

    bottom_score = np.zeros(n)
    acc_ok = np.zeros(n)
    combined = np.zeros(n)

    for i in range(window, n):
        if np.isnan(ma[i]) or ma[i] <= 0:
            continue
        # 偏离率：正=低于MA60，负=高于MA60
        dev = (ma[i] - close[i]) / ma[i]
        # 评分：低于MA60得高分，超过MA60+20%得0分
        if close[i] <= ma[i] * 1.20:
            # 从MA60*1.20到MA60*0.70映射为0到100
            bottom_score[i] = max(0.0, min(100.0, 100.0 * (1.0 - (close[i] - ma[i]*0.70) / (ma[i]*0.50))))
        else:
            bottom_score[i] = 0.0

        acc_ok[i] = acc[i]
        if bottom_score[i] >= 60 and acc[i] > 0.5:
            combined[i] = 1.0

    return bottom_score, acc_ok, combined
