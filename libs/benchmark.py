#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import time
from contextlib import contextmanager
from logging import Logger
from typing import Optional, Generator

from typeguard import typechecked



@contextmanager
@typechecked
def benchmark(
    operation_name: Optional[str] = None,
    log: Optional[Logger] = None,
) -> Generator[None, None, None]:
    """
    Function to benchmark a function
    """
    start: float = time.perf_counter()
    try:
        yield
    finally:
        elapsed: float = time.perf_counter() - start
        operation_name: str = operation_name or ""
        if log:
            log.debug(f"{operation_name} executed in {elapsed:.3f} seconds.")