from __future__ import annotations
import time
import contextlib

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
    def __enter__(self): self.st = time.perf_counter_ns()
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))