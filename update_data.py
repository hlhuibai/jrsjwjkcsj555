"""
每日数据更新脚本
用法: python update_data.py
可由 Windows 任务计划 / cron 在交易日 15:30 后自动执行
"""
import os, sys, json, datetime

sys.path.insert(0, os.path.dirname(__file__))

from engine import update_daily_data, _CACHE_DIR, FULL_UNIVERSE, BENCHMARK_INDICES

LOG_FILE = os.path.join(_CACHE_DIR, "update_log.txt")


def main():
    today = datetime.date.today()
    # 周末跳过（周六=5, 周日=6）
    if today.weekday() >= 5:
        print(f"[{today}] 周末，跳过更新。")
        return

    print(f"[{today}] 开始增量更新 {len(FULL_UNIVERSE)} 只股票 + {len(BENCHMARK_INDICES)} 个指数...")
    result = update_daily_data()

    # 写日志
    log_entry = {
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **result,
    }
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  日志写入失败: {e}")

    print(f"  成功: {result['updated']}, 失败: {result['failed']}")
    if result["failed_list"]:
        print(f"  失败列表: {result['failed_list']}")
    print(f"  日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
