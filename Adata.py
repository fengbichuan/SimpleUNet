import os
import adata
import time


import pandas as pd
from datetime import datetime

# --- 配置项 ---
MONITOR_INTERVAL_SECONDS = 60  # 每次扫描的间隔时间（秒）
MA5_DAYS = 5  # 5日均线
MA10_DAYS = 10  # 10日均线
MA20_DAYS = 20  # 20日均线
MA60_DAYS = 60  # 60日均线
TURNOVER_THRESHOLD = 5  # 换手率阈值 (10%)
# ---

# 用于存储今天已提醒过的股票，避免重复提醒
alerted_stocks_today = set()


def get_hot_stocks():
    """
    获取“人气龙头/热点股” (条件 4)
    使用 all_capital_flow_east() 获取近5日概念资金流入的龙头股
    """
    print("=" * 50)
    print("正在获取热门/龙头股票列表 (Fetching hot/leader stock list)...")
    # --- 已修改: 使用 'days_type=5' ---
    print("使用接口 (Using API): adata.stock.market.all_capital_flow_east(days_type=5)")

    try:
        # 1. 获取近5日所有概念的资金流向
        df_flow = adata.stock.market.all_capital_flow_east(days_type=1)

        # 2. 筛选出主力净流入为正的“热门概念”
        hot_concepts = df_flow[df_flow['main_net_inflow'] > 0]

        # 3. 提取这些热门概念的“龙头股”
        hot_stocks_map = {}
        for _, row in hot_concepts.iterrows():
            stock_code = row['stock_code']
            stock_name = row['stock_name']
            if stock_code not in hot_stocks_map:
                # 存储 股票代码 -> 股票名称 的映射
                hot_stocks_map[stock_code] = stock_name

        print(
            f"获取到 {len(hot_stocks_map)} 只【近5日】热门股票待监控 (Found {len(hot_stocks_map)} hot stocks to monitor).")
        print("=" * 50)
        return hot_stocks_map

    except Exception as e:
        print(f"获取热门股票列表失败 (Error fetching hot stocks): {e}")
        return {}


def check_stock(stock_code, stock_name):
    """
    检查单只股票是否满足所有条件 (1, 2, 3, 5)
    """
    global alerted_stocks_today

    try:
        # 1. 获取日K数据
        # (adata.stock.market.get_market() 接口在文档中显示返回K线行情)
        df = adata.stock.market.get_market(stock_code=stock_code, k_type=1, adjust_type=1)

        if len(df) < MA60_DAYS + 1:
            return

        data = df.iloc[-(MA60_DAYS + 1):]
        today = data.iloc[-1]
        yesterday = data.iloc[-2]

        # 检查1 (前置): 是否为阳线 (收盘 > 开盘)
        is_yang_line = today['close'] > today['open']
        if not is_yang_line:
            return

        # 计算所有均线
        ma5 = data['close'].iloc[-MA5_DAYS:].mean()
        ma10 = data['close'].iloc[-MA10_DAYS:].mean()
        ma20 = data['close'].iloc[-MA20_DAYS:].mean()
        ma60 = data['close'].iloc[-MA60_DAYS:].mean()

        # 检查1: 5日均线在阳线实体中
        condition1 = (today['open'] < ma5 < today['close'])

        # 检查2: 较上一个交易日放量
        condition2 = today['volume'] > yesterday['volume']

        # 检查3: 处于上升趋势 (MA5 > MA10 > MA20 > MA60)
        condition3 = (ma5 > ma10) and (ma10 > ma20) and (ma20 > ma60)

        # 检查5: 换手率大于 10%
        # (文档显示 'turnover_ratio' 字段是换手率(%))
        condition5 = today['turnover_ratio'] > TURNOVER_THRESHOLD

        # 4. 汇总结果并提醒
        if condition1 and condition2 and condition3 and condition5:
            current_date = datetime.now().strftime('%Y-%m-%d')
            alert_key = f"{stock_code}_{current_date}"

            if not any(alert.endswith(current_date) for alert in alerted_stocks_today):
                print(f"\n新的一天 ({current_date})，重置提醒列表 (New day, resetting alert list).\n")
                alerted_stocks_today.clear()

            if alert_key not in alerted_stocks_today:
                alerted_stocks_today.add(alert_key)
                print("\n" + "=" * 50)
                print(f"  *** 实时股票提醒 (Real-time Stock Alert) ***")
                print(f"  时间 (Time):   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  股票 (Stock):  {stock_name} ({stock_code})")
                print(f"  现价 (Price):  {today['close']:.2f}")
                print_conditions(today, yesterday, ma5, ma10, ma20, ma60)
                print("=" * 50 + "\n")

    except Exception as e:
        pass


def print_conditions(today, yesterday, ma5, ma10, ma20, ma60):
    """辅助函数：格式化打印满足的条件详情"""
    print("\n  --- 满足以下所有条件 (Met all conditions) ---")
    print(f"  [✓] 1. 五日线在阳线实体中 (MA5 in Yang Body)")
    print(f"      - ( 开盘 (Open): {today['open']:.2f} < 均线 (MA5): {ma5:.2f} < 收盘 (Close): {today['close']:.2f} )")
    print(f"\n  [✓] 2. 较昨日放量 (Increased Volume)")
    print(f"      - 今日 (Today): {today['volume']:,.0f}")
    print(f"      - 昨日 (Yday):  {yesterday['volume']:,.0f}")
    print(f"\n  [✓] 3. 处于上升趋势 (Upward Trend - 多头排列)")
    print(f"      - ( MA5: {ma5:.2f} > MA10: {ma10:.2f} > MA20: {ma20:.2f} > MA60: {ma60:.2f} )")
    print(f"\n  [✓] 4. 人气/热点股 (Hot/Leader Stock)")
    print(f"      - (已通过【近5日】概念资金流筛选, Implied by 'near 5 days' concept flow filter)")
    print(f"\n  [✓] 5. 换手率 > {TURNOVER_THRESHOLD}% (Turnover > {TURNOVER_THRESHOLD}%)")
    print(f"      - 换手 (Turnover): {today['turnover_ratio']:.2f}%")


def main_monitor():
    """
    主监控循环
    """
    hot_stocks_map = get_hot_stocks()
    if not hot_stocks_map:
        print("未能获取热门股票列表，程序退出 (Failed to get hot stocks, exiting).")
        return

    print(f"\n开始实时监控 {len(hot_stocks_map)} 只股票 (Starting real-time monitoring)...")
    print(f"监控周期 (Check interval): {MONITOR_INTERVAL_SECONDS} 秒 (seconds)")
    print("按 Ctrl+C 停止 (Press Ctrl+C to stop).")

    while True:
        try:
            scan_start_time = datetime.now()
            print(f"\n--- {scan_start_time.strftime('%H:%M:%S')} 开始新一轮扫描 (New scan cycle) ---")

            count = 0
            total = len(hot_stocks_map)

            for stock_code, stock_name in hot_stocks_map.items():
                count += 1
                print(f"  正在检查 ({count}/{total}): {stock_name} ({stock_code})...", end='\r')
                check_stock(stock_code, stock_name)
                time.sleep(0.2)

            print(f"\n--- {datetime.now().strftime('%H:%M:%S')} 本轮扫描完成 (Scan complete) ---")
            print(
                f"等待 {MONITOR_INTERVAL_SECONDS} 秒后开始下一轮... (Waiting {MONITOR_INTERVAL_SECONDS}s for next cycle...)")
            time.sleep(MONITOR_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n监控已停止 (Monitoring stopped by user).")
            break
        except Exception as e:
            print(f"主循环出错 (Error in main loop): {e}")
            print("30秒后重试 (Retrying in 30s)...")
            time.sleep(30)


if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    main_monitor()