import time
from contextlib import contextmanager
from functools import wraps


@contextmanager
def timer(label: str):
    print(f"starting: \"{label.upper().replace(" ", "_")}\" ...")
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"finished: \"{label.upper().replace(" ", "_")}\" ({dt:.3f}s)")


def timed(label: str | None = None):
    def decorator(func):
        name = label or func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    exit()