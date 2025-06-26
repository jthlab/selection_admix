import time
from functools import wraps
from typing import Any, Callable
import os


def timer(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = fn(*args, **kwargs)
        # return result
        end = time.time()
        if os.environ.get("BMWS_TIMER"):
            print(f"{fn.__name__} ran in {end - start:.4f} seconds")
        return result

    return wrapper
