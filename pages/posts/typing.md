---
title: Distributed Systems, Lazy Evaluation, and Static Typing
date: 2023/04/20
description: A detailed guide to static typing lazily evaluated python code
tag: python, mypy
author: You
---

# Brief

There is a common pattern across certain libraries (e.g prefect, dagster, airflow...) that enable code execution against distributed computing frameworks. The pattern is as follows:

Given a python function
```python
def add_one(x: int) -> int:
    return x + 1
```

we turn it into a lazily evaluated function by decorating it with something like a `@task` decorator
```python
@task
def add_one(x: int) -> int:
    return x + 1
```

The lazily evaluated function will then be evaluated when the entire "task graph" is submitted to a distributed computing framework like dask or ray. 

The problem is that the type signature of the lazily evaluated function is not the same as the original function. The lazily evaluated function will have a signature like
```python
def add_one(x: int) -> Future[int]:
    return x + 1
```

Where `Future` is a type that represents a value that will be available in the future. This is a problem because it means that we can't use the original function in a type checked way.

We rely on on the following three typing features to solve this problem:
1. ParamSpec (PEP 612) - allows decorators to preserve the type signature of the original function
2. Protocol (PEP 544) - allows us to define a protocol for the `Future` type 
3. Generic (PEP 484) - allows us to define a generic type that can be used with the `Future` protocol


```python
from typing import Callable, TypeVar, Protocol

In = ParamSpec("In")
Out = TypeVar("Out", covariant=True)

class TaskT(Generic[In, Out], Protocol):
    """Task Generic Protocol."""

    def __call__(self, *args: In.args, **kwds: In.kwargs) -> Out:
        """Returns a 'future-like' object that should be treated as Out by mypy."""

    def run(self, *args: In.args, **kwds: In.kwargs) -> Out:
        """Runs the underlying function and returns the result."""


def task(fn: Callable[In, Out]) -> TaskT[In, Out]:
    return create_task_from_function(fn)
```

Note this is simplified given in most decorator use-cases, one would want to allow both
```python
@task
def add_one(x: int) -> int:
    return x + 1
```
and
```python
@task()
def add_one(x: int) -> int:
    return x + 1
```
to be valid. This can be achieved by using the `@overload` decorator to define the two different signatures.


```python
# When we use the decorator without arguments (i.e @task), we directly get back
# a TaskT[In, Out] instance.
@overload
def task(
    fn: Callable[In, Out],
) -> TaskT[In, Out]:
    """Return a decorator that creates a task from a function."""


@overload
def task(
    fn: Literal[None] = None,
) -> Callable[[Callable[In, Out]], TaskT[In, Out]]:
    """Return a decorator that creates a task from a function."""


def task(
    fn: Optional[Callable[In, Out]] = None,
) -> Union[TaskT[In, Out], Callable[[Callable[In, Out]], TaskT[In, Out]]]:
```

