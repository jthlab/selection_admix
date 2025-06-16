import time
from functools import wraps
from typing import Any, Callable


def timer(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = fn(*args, **kwargs)
        # return result
        end = time.time()
        print(f"{fn.__name__} ran in {end - start:.4f} seconds")
        return result

    return wrapper
