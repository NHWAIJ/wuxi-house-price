# -*- coding: utf-8 -*-
"""
progress_gui.py — 运行进度弹窗(纯标准库 tkinter + threading,零依赖)。

线程安全设计(解决训练时窗口"未响应"问题):
  - 训练/预测等耗时计算在**子线程**执行;
  - tkinter 主线程 mainloop + after(100ms) 轮询队列刷新窗口;
  - 窗口始终可拖动/关闭,进度条平滑更新,不会出现 Windows"未响应"。

用法:
    from progress_gui import run_with_progress
    run_with_progress(_run, _make_progress)
    其中 _run(pw) 为耗时主流程(pw 可 None=纯控制台模式),内部调用 pw.update/done。
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk


class ProgressGUI:
    def __init__(self, title="运行中"):
        try:
            self.root = tk.Tk()
        except Exception as e:
            raise RuntimeError(f"无法创建图形窗口: {e}")
        self.root.title(title)
        self.root.geometry("600x440")
        self.root.minsize(480, 360)
        self._queue = queue.Queue()
        self._closed = False

        self.step_var = tk.StringVar(value="准备中…")
        tk.Label(self.root, textvariable=self.step_var,
                 font=("Microsoft YaHei", 11, "bold"),
                 anchor="w", fg="#0b0b0b").pack(fill="x", padx=14, pady=(12, 6))

        self.bar = ttk.Progressbar(self.root, maximum=100)
        self.bar.pack(fill="x", padx=14, pady=2)

        self.msg_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.msg_var,
                 font=("Microsoft YaHei", 9), anchor="w",
                 fg="#52514e").pack(fill="x", padx=14, pady=(4, 2))

        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=14, pady=8)
        self.log = tk.Text(frame, height=16, font=("Microsoft YaHei", 9),
                           state="disabled", wrap="word", bg="#fcfcfb",
                           relief="solid", borderwidth=1)
        sb = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        self._btn = None
        self.root.after(100, self._poll)     # 主线程轮询队列(子线程安全更新)
        self.root.update()

    # ---- 子线程可安全调用(入队,不阻塞计算) ----
    def update(self, text, pct, log=None):
        self._queue.put(("update", text, pct, log))

    def done(self, summary_lines):
        self._queue.put(("done", summary_lines))

    def _poll(self):
        """主线程刷新:处理队列消息。"""
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "update":
                    _, text, pct, log = item
                    self.step_var.set(text)
                    self.msg_var.set(f"进度 {pct:.0f}%")
                    self.bar["value"] = max(0, min(100, pct))
                    if log:
                        self._append(log)
                elif kind == "done":
                    _, lines = item
                    self.step_var.set("✅ 全部完成")
                    self.msg_var.set("进度 100%")
                    self.bar["value"] = 100
                    for ln in lines:
                        self._append("· " + ln)
                    if self._btn is None:
                        self._btn = tk.Button(self.root, text="完成,关闭窗口",
                                              command=self.root.destroy,
                                              font=("Microsoft YaHei", 10))
                        self._btn.pack(pady=8)
        except queue.Empty:
            pass
        if not self._closed:
            self.root.after(100, self._poll)

    def _append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self):
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass


def run_with_progress(target, make_progress, title="运行中"):
    """
    统一入口:耗时主流程 target(pw) 在子线程运行,主线程负责窗口刷新。
      - 有桌面:弹出进度窗口,计算在子线程(窗口始终响应,不"未响应");
      - 无桌面/BK_NO_GUI=1:纯控制台模式,直接运行 target(None)。
    target 内部调用 pw.update(text, pct, log) / pw.done(lines)(线程安全)。
    """
    pw = None
    try:
        pw = make_progress() if make_progress else None
    except Exception as e:
        print(f"[提示] 进度窗口创建失败({e}),改用控制台输出")
        pw = None

    if pw is None:
        target(None)
        return

    def _worker():
        try:
            target(pw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                pw.update(f"运行出错: {e}", 100, log=str(e))
                pw.done(["请查看控制台完整报错信息"])
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()
    pw.root.mainloop()
