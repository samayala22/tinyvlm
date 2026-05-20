from __future__ import annotations
from functools import wraps
import time
import contextlib
import os

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
    def __enter__(self): self.st = time.perf_counter_ns()
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.perf_counter()
            exec_time = end-start
            print(f"*** {func.__name__}: {exec_time:.3f} s")
    return _time_it

def getenv(key):
    var = os.getenv(key)
    if not var or int(var) == 0:
        return False
    return True