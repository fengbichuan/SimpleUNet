import sys
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
# [!!] 修正: 移除未使用的 tqdm 导入
# from tqdm import tqdm

# --- PyQt5 Imports [!! 已修改] ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QProgressBar, QTableWidget,
    QLabel, QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot as Slot, Qt

warnings.simplefilter(action='ignore', category=FutureWarning)


# -----------------------------------------------------------------
# 1. 工作线程 (Worker)
# -----------------------------------------------------------------
# QObject必须是所有工作类的基类，以便它可以被移动到QThread
class Worker(QObject):
    """
    工作线程，用于处理所有耗时的Tushare API请求和数据分析
    """
    # --- 定义信号 ---
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    found = pyqtSignal(str, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.pro = None
        self.is_running = False

    def run_scan(self, token):
        """
        主扫描逻辑，这是在QThread中执行的函数
        """
        if self.is_running:
            return

        self.is_running = True

        try:
            # -----------------------------------------------------------------
            # 1. 初始化 Tushare
            # -----------------------------------------------------------------
            self.log.emit("正在初始化Tushare...")
            if not token:
                raise Exception("Tushare Token 不能为空！")

            ts.set_token(token)
            self.pro = ts.pro_api()
            # 尝试连接
            self.pro.trade_cal(limit=1)
            self.log.emit("Tushare 连接成功。")

            # -----------------------------------------------------------------
            # 2. 获取交易日期
            # -----------------------------------------------------------------
            self.log.emit("正在获取交易日期...")
            T_1_DATE, T_2_DATE = self.get_trade_dates()
            self.log.emit(f"目标扫描日期 (T-1): {T_1_DATE}")
            self.log.emit(f"成交量对比日期 (T-2): {T_2_DATE}")

            # -----------------------------------------------------------------
            # 3. 获取待扫描的股票列表
            # -----------------------------------------------------------------
            self.log.emit("正在获取待扫描股票列表...")
            stock_dict = self.get_scan_list(T_1_DATE)
            if not stock_dict:
                raise Exception("没有获取到股票列表，程序退出。")

            total_stocks = len(stock_dict)
            self.log.emit(f"\n--- 开始扫描 {total_stocks} 只股票 ---")

            # -----------------------------------------------------------------
            # 4. 循环扫描
            # -----------------------------------------------------------------
            results_list = []
            start_time = time.time()

            for i, (code, name) in enumerate(stock_dict.items()):
                if not self.is_running:  # 允许外部停止
                    self.log.emit("扫描被手动中止。")
                    break

                # Tushare Pro 有调用频率限制
                time.sleep(0.2)  # 保持 0.2 秒延迟 (安全起见)

                # 检查股票
                match = self.check_stock(code, T_1_DATE)

                if match:
                    results_list.append((code, name))
                    # 发送“找到”信号
                    self.found.emit(code, name)
                    self.log.emit(f"  >> 找到匹配: {code} ({name})")

                # 更新进度条
                progress_percent = int((i + 1) * 100 / total_stocks)
                self.progress.emit(progress_percent)

            # -----------------------------------------------------------------
            # 5. 输出结果
            # -----------------------------------------------------------------
            end_time = time.time()
            self.log.emit("\n--- 扫描完成 ---")
            self.log.emit(f"总耗时: {end_time - start_time:.2f} 秒")

            if results_list:
                final_msg = f"扫描完成：共找到 {len(results_list)} 只满足条件的股票。"
            else:
                final_msg = "扫描完成：未找到满足所有条件的股票。"

            self.log.emit(final_msg)
            self.finished.emit(final_msg)

        except Exception as e:
            error_msg = f"程序运行出错: {e}"
            self.log.emit(error_msg)
            self.error.emit(error_msg)
        finally:
            self.is_running = False

    def stop_scan(self):
        """
        外部调用的停止方法
        """
        self.is_running = False

    # -----------------------------------------------------------------
    # 以下是你原来的辅助函数，现在作为Worker类的方法
    # -----------------------------------------------------------------

    def get_trade_dates(self):
        """
        获取最近的两个交易日 (T-1 和 T-2)
        """
        today_str = datetime.now().strftime('%Y%m%d')
        start_date_str = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')

        trade_cal = self.pro.trade_cal(exchange='', start_date=start_date_str, end_date=today_str)
        open_days = trade_cal[trade_cal['is_open'] == 1]
        recent_open_days = open_days[open_days['cal_date'] < today_str]

        if len(recent_open_days) < 2:
            raise Exception("无法获取足够的交易日数据，请检查Tushare连接或是否处于长假期间。")

        recent_trade_days = recent_open_days.head(2)
        T_1_DATE = recent_trade_days.iloc[0]['cal_date']
        T_2_DATE = recent_trade_days.iloc[1]['cal_date']

        return T_1_DATE, T_2_DATE

    def get_scan_list(self, trade_date):
        """
        获取待扫描的股票列表 (同花顺人气榜或全部A股)
        """
        self.log.emit("尝试获取同花顺人气榜...")
        try:
            df_hot = self.pro.ths_hot(trade_date=trade_date, type='R', fields='ts_code,name')
            if not df_hot.empty:
                self.log.emit(f"成功获取 {len(df_hot)} 只同花顺人气榜股票。")
                return dict(zip(df_hot.ts_code, df_hot.name))
            else:
                self.log.emit("ths_hot 接口未返回数据。")
        except Exception as e:
            self.log.emit(f"无法获取 'ths_hot' (Tushare积分不足或API异常): {e}")

        self.log.emit("回退策略：获取所有A股列表进行扫描。")
        df_all = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        df_all = df_all[~df_all['name'].str.contains('ST')]
        self.log.emit(f"将扫描 {len(df_all)} 只非ST股票。")
        return dict(zip(df_all.ts_code, df_all.name))

    def check_stock(self, ts_code, end_date):
        """
        检查单个股票是否满足所有筛选条件 (不复权)
        """
        try:
            # a. 获取 "不复权" 日线行情数据
            df_daily = self.pro.daily(ts_code=ts_code, end_date=end_date, limit=100)

            if df_daily.empty or len(df_daily) < 61:
                return None

            # b. 手动计算均线
            df_daily = df_daily.iloc[::-1]  # 升序
            df_daily['ma5'] = df_daily['close'].rolling(window=5).mean()
            df_daily['ma10'] = df_daily['close'].rolling(window=10).mean()
            df_daily['ma20'] = df_daily['close'].rolling(window=20).mean()
            df_daily['ma60'] = df_daily['close'].rolling(window=60).mean()
            df_daily = df_daily.iloc[::-1]  # 降序

            # c. 获取每日基本指标 (换手率)
            df_basic = self.pro.daily_basic(ts_code=ts_code, trade_date=end_date, fields='ts_code,turnover_rate')

            if df_basic.empty:
                return None

            t1 = df_daily.iloc[0]
            t2 = df_daily.iloc[1]
            t1_basic = df_basic.iloc[0]

            if pd.isna(t1['ma60']) or pd.isna(t1['ma5']):
                return None

            # --- 开始逐条检查 ---

            # 条件1: T-1 (昨天) 必须是阳线
            if not (t1['close'] > t1['open']):
                return None

            # 条件1: 五日均线在昨日阳线的实体中
            if not ((t1['ma5'] >= t1['open']) and (t1['ma5'] <= t1['close'])):
                return None

            # 条件2: 较上一个交易日放量
            if not (t1['vol'] > t2['vol']):
                return None

            # 条件3: 处于上升趋势中 (多头排列)
            if not ((t1['ma5'] > t1['ma10']) and \
                    (t1['ma10'] > t1['ma20']) and \
                    (t1['ma20'] > t1['ma60'])):
                return None

            # 条件5: 换手率大于10%
            if not (t1_basic['turnover_rate'] > 10):
                return None

            # --- 所有条件均满足 ---
            return ts_code

        except Exception as e:
            # 记录轻微错误，但继续运行
            # self.log.emit(f"  [Warn] 处理 {ts_code} 时出错: {e}")
            return None


# -----------------------------------------------------------------
# 2. 主窗口 (GUI)
# -----------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tushare 股票筛选器 (PyQt5版)")  # [!! 已修改]
        self.setGeometry(100, 100, 800, 600)

        # --- 初始化UI ---
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 1. Token 输入行
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Tushare Token:"))
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("请在这里输入你的Tushare Pro Token")
        # ！！将你脚本中的Token作为默认值
        self.token_input.setText("33c189692cf25347cfacb0c27104163a283d9741ff97dd517f13d25b")
        token_layout.addWidget(self.token_input)

        self.start_button = QPushButton("开始扫描")
        self.start_button.clicked.connect(self.start_scan)
        token_layout.addWidget(self.start_button)

        main_layout.addLayout(token_layout)

        # 2. 结果表格
        main_layout.addWidget(QLabel("扫描结果:"))
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["股票代码", "股票名称"])
        # [!! 已修改] PyQt5 使用 QHeaderView.Stretch
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # [!! 已修改] PyQt5 使用 QTableWidget.NoEditTriggers
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 禁止编辑
        main_layout.addWidget(self.results_table)

        # 3. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # 4. 日志输出
        main_layout.addWidget(QLabel("实时日志:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(150)
        main_layout.addWidget(self.log_output)

        # --- 线程设置 ---
        self.thread = None
        self.worker = None

    def start_scan(self):
        """
        点击“开始扫描”按钮时触发
        """
        if self.worker and self.worker.is_running:
            # 如果正在运行，按钮变为"停止"
            self.worker.stop_scan()
            self.start_button.setText("正在停止...")
            self.start_button.setEnabled(False)
            return

        # --- 重置UI ---
        self.start_button.setText("正在扫描... (点击停止)")
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.results_table.setRowCount(0)  # 清空表格

        # --- 创建并启动线程 ---
        self.thread = QThread()
        self.worker = Worker()

        token = self.token_input.text()

        # 1. 将 worker 移动到 thread
        self.worker.moveToThread(self.thread)

        # 2. 连接信号和槽
        #    当线程启动时，调用 worker.run_scan
        self.thread.started.connect(lambda: self.worker.run_scan(token))

        #    连接 worker 的信号到 GUI 的槽函数
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.set_progress)
        self.worker.found.connect(self.add_stock_to_table)
        self.worker.finished.connect(self.scan_finished)
        self.worker.error.connect(self.scan_error)

        #    扫描完成后，自动清理线程
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # 3. 启动线程
        self.thread.start()

    # --- GUI 槽函数 ---

    @Slot(str)  # 显式声明为槽
    def append_log(self, message):
        """将日志追加到 QTextEdit"""
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    @Slot(int)
    def set_progress(self, value):
        """设置进度条的值"""
        self.progress_bar.setValue(value)

    @Slot(str, str)
    def add_stock_to_table(self, code, name):
        """向表格中添加一行数据"""
        row_count = self.results_table.rowCount()
        self.results_table.insertRow(row_count)
        self.results_table.setItem(row_count, 0, QTableWidgetItem(code))
        self.results_table.setItem(row_count, 1, QTableWidgetItem(name))

    def scan_finished(self, final_message):
        """扫描正常完成时调用"""
        self.append_log(f"--- {final_message} ---")
        self.progress_bar.setValue(100)
        self.start_button.setText("开始扫描")
        self.start_button.setEnabled(True)
        self.worker = None  # 清理
        self.thread = None

    def scan_error(self, error_message):
        """扫描中发生严重错误时调用"""
        self.scan_finished("扫描因错误中止。")
        # 弹窗显示严重错误
        QMessageBox.critical(self, "扫描出错", error_message)

    def closeEvent(self, event):
        """关闭窗口时，确保停止工作线程"""
        if self.worker and self.worker.is_running:
            self.worker.stop_scan()
        event.accept()


# -----------------------------------------------------------------
# 3. 主程序入口
# -----------------------------------------------------------------
if __name__ == "__main__":
    # 适配高DPI屏幕 (PyQt5 方式) [!! 已修改]
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # [!! 已修改] PyQt5 使用 app.exec_()
    sys.exit(app.exec_())

