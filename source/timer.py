import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    print(f"starting: \"{label.upper().replace(" ", "_")}\" ...")
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"finished: \"{label.upper().replace(" ", "_")}\" ({dt:.3f}s)")


if __name__ == "__main__":
    exit()