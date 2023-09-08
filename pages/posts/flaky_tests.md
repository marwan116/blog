---
title: "How not to footgun yourself when writing tests - a showcase of flaky tests"
date: 2023/09/05
description: Hard-earned learnings about flaky tests inspired by a talk-python podcast titled "Taming Flaky Tests"
tag: testing, python, pytest
author: Marwan
---

## Intro
I am writing this article after listening to a [talk python](https://talkpython.fm/) podcast episode titled "Taming Flaky Tests" where Michael Kennedy, the host of the podcast, interviewed [Gregory M. Kapfhammer](https://www.gregorykapfhammer.com/) and [Owain Parry](https://www.linkedin.com/in/owain-parry-a0040a216) to discuss their research on software testing.

My hope is to supplement the podcast by sharing examples of flaky tests inspired by tests I have encountered in the wild and by providing tips on how to detect and avoid flakiness. Note that all the code shared in this article can be found on the associated github repo [flakycode](https://www.github.com/marwan116/flakycode)

## Outline

I outline the different types of flaky tests by borrowing a categorization from the paper discussed in the podcast [Surveying the developer experience of flaky tests](https://www.gregorykapfhammer.com/download/research/papers/key/Parry2022-paper.pdf).

- [Definition of flaky tests](#definition-of-flaky-tests)
- [Intra-test flakiness](#intra-test-flakiness)
    - [Concurrency - The GIL won't save you](#concurrency---the-gil-wont-save-you)
    - [Randomness](#randomness)
        - [Algorithmic non-determinism](#algorithmic-non-determinism--careful-with-tolerance-values)
    - [Floating point arithmetic](#floating-point-arithmetic)
        - [Underflow or overflow issues](#underflow-or-overflow-issues)
        - [Loss in precision](#loss-in-precision)
    - [Missing corner cases (test is too restrictive)](#missing-corner-cases-test-is-too-restrictive)
        - [Fuzzing to find corner cases](#fuzzing-to-find-corner-cases)
    - [Timeout - Make sure your timeout is not too short](#timeout---make-sure-your-timeout-is-not-too-short)
- [Inter-test flakiness](#inter-test-flakiness)
    - [Test order dependency](#test-order-dependency)
        - [State pollution by incorrect mocking/monkey-patching](#state-pollution-by-incorrect-mockingmonkey-patching)
        - [Database state pollution](#database-state-pollution)
- [External factors](#external-factors)
    - [Network](#network)
    - [Asynchronous wait](#asynchronous-waiting-on-external-resources)

## Definition of flaky tests
A flaky test as defined by [pytest](https://docs.pytest.org/en/6.2.x/flaky.html) is one that exhibits intermittent or sporadic failure, that seems to have non-deterministic behaviour. Sometimes it passes, sometimes it fails, and it’s not clear why.


## Intra-Test Flakiness 
We will begin by discussing intra-test flakiness. This type of flakiness stems from how the test is implemented. It is not due to interference from other tests or from external factors, such as network or file system disturbances.

### Concurrency - The GIL won't save you

First up is an example of a test where I attempt to make use of concurrency to speed up the test's runtime only to shoot myself in the foot by introducing flakiness.

Can you spot the problem with this test example? (Hint: always think twice before reaching out to the `threading` module to speed up your test suite.)

```python
# ----------------------------------
# implementation - bank_account.py
# ----------------------------------
class BankAccount:
    def __init__(self, balance=100):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount


class Merchant:
    def charge_and_refund(
        self,
        account,
        test_fee=0.01,
        num_transactions=400
    ):
        for _ in range(num_transactions):
            account.withdraw(test_fee)
            account.deposit(test_fee)

# ----------------------------------
# test suite - test_bank_account.py
# ----------------------------------
import threading

def test_charge_and_refund_keeps_balance_the_same():
    # initialize bank account with $100 balance
    account = BankAccount(balance=100)
    original_balance = account.balance

    threads = []

    # "smartly" parallelize call to charge_and_refund
    merchants = [Merchant() for _ in range(10)]
    for merchant in merchants:
        thread = threading.Thread(
            target=merchant.charge_and_refund,
            args=(account,),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert account.balance == original_balance
```

The problem in this test arises because the `BankAccount` implementation is not thread-safe. The balance attribute isn't protected by a lock, allowing multiple threads to access and modify it simultaneously.

But wait a minute, shouldn't the GIL (Global Interpreter Lock) save us from this concurrency issue? The GIL contrary to "popular fallacy" does not provide atomicity or synchronization guarantees for complex operations involving multiple bytecode instructions. What I mean here is that the GIL will cause threads to interleave but they can still interleave in a way that one thread runs `deposit` while another thread runs `withdraw` at the same time.

To ensure thread safety in scenarios like this, you would need to use proper synchronization mechanisms like locks and sempahores to protect shared state in your code, ensuring that only one thread can modify shared data at a time.

**Tips to detect this kind of flakiness** if you want to increase the likelihood of your test suite catching concurrency issues, it is recommended you:
- set a smaller switch interval to force the python interpreter to switch between threads more frequently
- run your test more than once

i.e. Your test suite should be running a code equivalent to the following:

```python
# test suite - test_bank_account.py
....

if __name__ == "__main__":
    original_switch_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.0001)
        passed = 0
        n_iter = 100
        for _ in range(n_iter):
            out = pytest.main([__file__])
            if out == pytest.ExitCode.OK:
                passed += 1
        print(
            "passed_percent",
            passed / n_iter * 100, "%"
        )
    finally:
        sys.setswitchinterval(original_switch_interval)
```

To avoid writing boilerplate code for running your test multiple times, you can use the [pytest-repeat](https://pypi.org/project/pytest-repeat/) plugin. Worth pointing out a similar and perhaps more straightforward plugin [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder) which is meant to find flaky tests by running your test suite multiple times and comparing the results.

Additionaly, you can create a pytest fixture to alter the switch interval of the test

```python
@pytest.fixture
def fast_switch_interval():
    original = sys.getswitchinterval()
    sys.setswitchinterval(0.0001)
    try:
        yield
    finally:
        sys.setswitchinterval(original)
```
Note however that you are modifying the switch interval for the entire test suite given the sys module is global... For more details on the switch interval, see this article from [superfastpython](https://superfastpython.com/context-switch-interval-in-python/#Why_Change_the_Switch_Interval)

### Randomness
Introducing randomness when constructing test inputs offers some benefits. It can help you test a wider range of inputs and avoid biasing your tests towards a specific input. However, it can also lead to flakiness if it is not properly handled.

#### Algorithmic non-determinism - careful with tolerance values
Randomness when combined with non-deterministic algorithms can lead to flakiness. This is because the output of the algorithm is not guaranteed to be the same given the same input. 

Here is a somewhat common example where I check the output of an optimization algorithm is within a reasonable range of the expected output. Can you spot the problem?

```python
# ----------------------------------
# implementation - minimze_rosenbrock.py
# ----------------------------------
import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    a = 1.0
    b = 100.0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def minimize_rosenbrock(initial_guess):
    return minimize(
        rosenbrock,
        initial_guess,
        method='Nelder-Mead'
    )

# ----------------------------------
# test suite - test_minimize_rosenbrock.py
# ----------------------------------

def test_correctly_minimizes_rosenbrock():
    # Initial guess
    initial_guess = np.random.randint(0, 10, size=2)

    # Get the result of the minimization
    result = minimize_rosenbrock(initial_guess)

    # naively choose atol
    naively_chosen_atol = 1e-5

    assert np.all(
        np.isclose(
            result.x, [1, 1], atol=naively_chosen_atol)
        )
```

This test is flaky because the chosen `Nelder-Mead` minimization algorithm is not always guaranteed to converge to the same true minimum - i.e. the algorithm is not deterministic.

More specifically, in this example I am using `Nelder-Mead` to minimize the `rosenbrock` function, also referred to as the Valley or Banana function, which is a popular test problem for minimization algorithms. The function is unimodal, and the global minimum lies in a narrow, parabolic valley. The narrow valley makes it somewhat difficult for the algorithms to converge exactly to the true minimum (1, 1) given different starting points.

In this case, the tester, knowing that the `Nelder-Mead` will only offer an approximate value of the true minimum, chose a "naive" tolerance value of 1e-5 without proper consideration to the expected variation in the result of the optimization algorithm.

I must admit it is hard not to succumb to the tendency to hand-wave tolerance/epsilon values. It usually goes something like this: "Well 1e-5 is a relatively large tolerance value compared to what I tried before and it now makes the test pass when I run it on my machine so it should be good enough".

Turns out the chosen tolerance value of 1e-5 makes the test flaky fail almost 65% of the time given the initial guess is randomly chosen from the range 0 to 10. Important to note here that while too small a tolerance value leads to false alarms, naively solving this issue by updating to a very large tolerance value means that the test is not sensitive enough to detect when the result is not within a statistically reasonable range.

To remedy this, we should assess the expected variance of the results of running the `Nelder-Mead` algorithm multiple times. Then we can chose an informed tolerance (in this example I take 3 standard deviations from the mean - tis way, we can be 99.7% confident, assuming the results are normally distributed, that the test will pass - not too permissive, not too strict). 

The updated test looks like this:

```python
....

def estimate_tolerance(num_runs=50):
    results = []

    for _ in range(num_runs):
        initial_guess = np.random.randint(
            0, 10, size=2
        )
        result = minimize_rosenbrock(initial_guess)
        results.append(result.x)

    results = np.array(results)
    std_dev = np.std(results)
    return 3 * std_dev

def test_correctly_minimizes_rosenbrock():
    # Initial guess
    initial_guess = np.random.randint(0, 10, size=2)

    # Get the result of the minimization
    result = minimize_rosenbrock(initial_guess)

    # tolerance is estimated from results of
    # running minimization multiple times
    tolerance = estimate_tolerance()

    assert np.all(
        np.isclose(
            result.x, [1, 1], atol=tolerance
        )
    )
```

### Floating point arithmetic

Dealing with floating point arithmetic can be tricky. 

On many occassions, I have resorted to using 32-bit data types like float32 and int32 to save memory especially when dealing with large arrays of data (i.e. in pandas or numpy). However, this can be a source of test flakiness if you are not careful.

Let's start with a tricky question:

True or false - guess the outcome?
```python
import numpy as np

x_64 = 262_143.015625
x_32 = np.float32(x_64 - 1)
x_64 - x_32 == 1
```
The outcome is True.

Very similar question - only very slightly different:
```python
import numpy as np

x_64 = 262_145.015625
x_32 = np.float32(x_64 - 1)
x_64 - x_32 == 1
```
The outcome is now False - because in laymen terms beyond 2^18 (262,144), the float32 data type can no longer maintain a precision of 1/64 (0.015625). To learn more about this, I highly recommend reading the article ["The problem with float32: you only get 16 million values" on pythonspeed.com](https://pythonspeed.com/articles/float64-float32-precision/)


#### Loss in precision

So how does this relate to test flakiness? Well, let's consider the following example and see if you can spot the problem:

```python
# ----------------------------------
# implementation - compute_balance.py
# ----------------------------------
import numpy as np


def compute_balance(amount, includes_flag):
    total_balance = np.float32(0)
    for amount, flag in zip(amount, includes_flag):
        if flag:
            total_balance += amount
    return total_balance


# ----------------------------------
# test suite - test_compute_balance.py
# ----------------------------------
def test_eng_balance_is_correctly_computed():
    dept_expense = 1_630_000_000
    num_dept = 10
    num_eng_dept = np.random.randint(1, num_dept)
    num_non_eng_dept = num_dept - num_eng_dept

    total_expenses = np.array(
        [dept_expense] * num_dept, dtype=np.float32
    )
    is_eng_dept = np.array(
        [True] * num_eng_dept +
        [False] * num_non_eng_dept,
        dtype="bool"
    )

    computed_total_eng_spend = compute_balance(
        total_expenses, is_eng_dept
    )
    expected_total_eng_spend = (
        dept_expense * num_eng_dept
    )
    diff = (
        computed_total_eng_spend -
        expected_total_eng_spend
    )
    assert np.isclose(diff, 0)
```

In this example, we build a test that checks if the total engineering spend computed by running `compute_balance` which sums up the balances of all engineering departments is equal to the total engineering spend computed by multiplying the department cost by the number of engineering departments.

This test is flaky due to a loss in precision when summing large numbers. To demonstrate the issue, I define a `precise_float32` function to check if a float32 will result in a significant loss of precision when compared to a float64.

```python
In [1]: import numpy as np

In [2]: 
    ...: def precise_float32(value):
    ...:     tol = np.finfo(np.float32).eps
    ...:     if np.abs(
    ...:        np.float32(value) - np.float64(value)
    ...:     ) > tol:
    ...:         raise ValueError("Loss of precision")
    ...:     return np.float32(value)
    ...: 

In [21]: precise_float32(value=1.63e9 * 2)
Out[21]: 3260000000.0

In [4]: precise_float32(value=1.63e9 * 3)
-------------------------------------------------------
ValueError            Traceback (most recent call last)
Cell In[4], line 1
----> 1 precise_float32(value=1.63e9 * 3)

Cell In[17], line 4, in precise_float32(value)
      2 tol = np.finfo(np.float32).eps
      3 if np.abs(
      4  np.float32(value) - np.float64(value)
      5  ) > tol:
----> 6     raise ValueError("Loss of precision")
      7 return np.float32(value)

ValueError: Loss of precision
```

**Tips to prevent this kind of flakiness**  To make an implementation more reliable, resort to division and subtraction to transform large number operations into smaller number operations to avoid loss of precision or overflow issues.

#### Underflow or overflow issues

Well you might be thinking, integers don't suffer from loss of precision issues. So why not use integers instead of floats? Well, you can still run into issues with integers if you are not careful. Consider the following example:

```python
# ----------------------------------
# implementation - compute_balance.py
# ----------------------------------
import numpy as np

def compute_balance(amount, includes_flag):
    total_balance = np.int32(0)
    for amount, flag in zip(amount, includes_flag):
        if flag:
            total_balance += amount
    return total_balance

# ----------------------------------
# test suite - test_compute_balance.py
# ----------------------------------
def test_eng_balance_is_correctly_computed():
    dept_expense = 1_630_000_000
    num_dept = 10
    num_eng_dept = np.random.randint(1, num_dept)
    num_non_eng_dept = num_dept - num_eng_dept

    total_expenses = np.array(
        [dept_expense] * num_dept, dtype=np.int32
    )
    is_eng_dept = np.array(
        [True] * num_eng_dept +
        [False] * num_non_eng_dept,
        dtype="bool"
    )

    computed_total_eng_spend = compute_balance(
        total_expenses, is_eng_dept
    )
    expected_total_eng_spend = (
        dept_expense * num_eng_dept
    )
    diff = (
        computed_total_eng_spend -
        expected_total_eng_spend
    )
    assert np.isclose(diff, 0)
```

We check if the total engineering budget computed by summing up the balances of all engineering departments is equal to the total engineering budget computed by multiplying the engineering department cost by the number of engineering departments.

This test is flaky due to an overflow in np.int32 - more specifically the maximum value for np.int32 is 2^31 (2,147,483,647) (one bit is used for the sign) - which can be verified by running the following code in a python interpreter:

```python
In [1]: import numpy as np

In [2]: np.iinfo(np.int32).max
Out[2]: 2147483647
```

Given 1 engineering department with a cost of 1.6 billion is still smaller than the max value of np.int32 ~2.1 billion, the test passes. However, given 2 engineering departments with a cost of 1.6 billion, the test fails due to an overflow in np.int32.

This can be fixed by making our implementation more robust to overflow either by using a larger data type like np.int64 or by forcing an overflow error to be raised when we run our test.

The latter can be achieved by using np.seterr to set the overflow error to be raised like so:

```python
def test_eng_balance_is_correctly_computed():
    np.seterr(over="raise")
    ...
```


### Missing corner cases (test is too restrictive)

Sometimes, a test is too restrictive and does not account for all possible corner cases. This can lead to flakiness.

Here I show some sample pandas code that attempts to stack and unstack a dataframe. Can you spot the problem?

```python
# ----------------------------------
# implementation - frame_stacker.py
# ----------------------------------
import pandas as pd

class FrameStacker:
    def stack(self, data):
        return data.stack()

    def unstack(self, data):
        return data.unstack()


# ----------------------------------
# test suite - test_frame_stacker.py
# ----------------------------------
import pytest
import numpy as np

@pytest.fixture
def data():
    """A way to mimick randomly generated data."""
    nrows = np.random.randint(0, 10)
    return pd.DataFrame({
        "A": np.random.choice(
            [1.0, 2.0, 3.0, np.nan], size=nrows
        ),
        "B": np.random.choice(
            [1.0, 2.0, 3.0, np.nan], size=nrows
        ),
    })

def test_stack_unstack_roundtrip(data):
    stacker = FrameStacker()
    stacked = stacker.stack(data)
    output = stacker.unstack(stacked)
    pd.testing.assert_frame_equal(output, data)
```

We have a test that checks if the output of unstacking the output of stacking a dataframe is equal to the original dataframe. This test is flaky because the implementation does not correctly account for a few corner cases - namely:
- columns "A" or "B" containing NaN values leads to data loss when stacking
- `nrows=0` (i.e. an empty dataframe) which leads to a conversion of the dataframe to a series when stacking

We can fix this by updating our implementation to account for these corner cases.

```python
sentinel_val = object()

class FrameStacker:

    def __init__(self, dtypes):
        self.dtypes = dtypes

    def stack(self, data):
        # handle case where data is empty
        if data.empty:
            return data
        # handle case where data contains nan values
        data = data.fillna(sentinel_val)
        # perform the stack
        return data.stack()

    def unstack(self, data):
        # handle case where data is empty
        if data.empty:
            return data
        # perform the unstack
        data = data.unstack()
        # replace sentinel values with nan
        data[data == sentinel_val] = np.nan
        # convert to original dtypes
        return data.astype(self.dtypes)
```

#### Fuzzing to find corner cases

One way to find the above-mentioned corner cases is to use fuzzing. Fuzzing is a technique where you intentionally generate many inputs to a function to try to find corner cases. One common approach to fuzzing is to use a property-based testing framework like [hypothesis](https://hypothesis.readthedocs.io/en/latest/).

Here is an example of how you can use hypothesis to find the corner cases mentioned above:

```python
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import (
    column, data_frames
)

@given(data=data_frames(
    columns=[
        column(
            name="A",
            dtype=float,
            elements=st.floats(allow_nan=True)
        ),
        column(
            name="B",
            dtype=float,
            elements=st.floats(allow_nan=True)
        ),
    ],
))
@settings(max_examples=100) 
def test_stack_unstack_roundtrip(data):
    stacker = FrameStacker(data.dtypes)
    stacked = stacker.stack(data)
    output = stacker.unstack(stacked)
    pd.testing.assert_frame_equal(output, data)
```

This test would fail most likely fail given 100 examples are usually enough to find the corner cases mentioned above (hypothesis is not exactly random, it is a bit smarter than that).

### Timeout - Make sure your timeout is not too short

Adding timeouts to tests can be a good way to ensure that tests do not take too long to run. However, if the timeout is too short, it can lead to flakiness. Consider the following example:

```python
# ----------------------------------
# implementation - find_discount_rate.py
# ----------------------------------
import numpy as np
from scipy.optimize import newton


def calculate_present_value(discount_rate, cashflows):
    t = np.arange(1, len(cashflows) + 1)
    return np.sum(cashflows / (1 + discount_rate) ** t)


def optimization_problem(discount_rate, present_value, cashflows):
    return calculate_present_value(discount_rate, cashflows) - present_value


def find_discount_rate(present_value, cashflows):
    try:
        return newton(
            optimization_problem, x0=0.1, args=(present_value, cashflows), maxiter=1000
        )
    except RuntimeError:
        # failed to converge
        return np.nan


# ----------------------------------
# test suite - test_find_discount_rate.py
# ----------------------------------
import pytest


@pytest.mark.timeout(2)
def test_find_discount_rate():
    h = 360_000
    cashflows = np.random.randint(50, 300, size=h)
    present_value = np.random.randint(1000, 100_000)
    find_discount_rate(present_value, cashflows)
```

This test module shows a flaky test due to an enforced test timeout that is usually good enough when the newton-rhapson algorithm converges quickly. However, when the algorithm fails to converge, the test will fail due to a timeout.

To resolve this, we can ensure that a smaller timeout is enforced on the optimization problem. We can do this by using the [stopit](https://pypi.org/project/stopit/) library to enforce a timeout on the optimization problem.

i.e. our implementation would look like this:

```python
import stopit

...

def find_discount_rate(present_value, cashflows):
    res = np.nan
    with stopit.ThreadingTimeout(seconds=1):
        try:
            res = newton(
                optimization_problem,
                x0=0.1,
                args=(present_value, cashflows),
            )
        except RuntimeError:
            # failed to converge
            res = np.nan
    return res
```

## Inter-Test Flakiness

Inter-test generally means flakiness stemming from the interference of other tests and not due to any inherent flakiness in a given test nor the interference of external factors like the network or the file system.

### Test order dependency

In general, tests should be independent of each other. However, sometimes test suites are improperly implemented leading to inter-test flakiness.

#### State pollution by incorrect mocking/monkey-patching

Mocking and monkey-patching are powerful tools that are often used when testing. However, if not used properly, they can lead to inter-test flakiness.

In the below example, we show how improper use of monkey-patching can lead to inter-test flakiness.

```python
# ----------------------------------
# implementation - my_service.py
# ----------------------------------
import datetime

class MyService:
    def get_current_time(self):
        return datetime.date.today()

# ----------------------------------
# test suite - test_my_service.py
# ----------------------------------
import datetime

class NewDate(datetime.date):
    @classmethod
    def today(cls):
        return cls(1990, 1, 1)


def test_we_are_back_in_the_90s():
    # let's monkey-patch datetime.date
    # what could go wrong?
    datetime.date = NewDate
    service = MyService()
    result = service.get_current_time()
    assert result.year == 1990

def test_we_are_in_the_21st_century():
    assert datetime.date.today().year >= 2000
```

In this example, we have a service that returns the current time. We have a test that checks if the current time is in the 90s. We also have another test that checks if the current time is in the 21st century.

The test `test_we_are_back_in_the_90s` introduces the flakiness because it incorrectly monkey-patches `datetime.date` to return January 1st 1990. This monkey-patching is not properly cleaned up after the test is run. As such, the test `test_we_are_in_the_21st_century` will fail because it will be run after the first test and it will be run in the 90s.

How is this flaky you might ask - if a developer runs this on their machine shouldn't they immediately get an error and fix it? Well, not necessarily. The two tests might be in separate test modules and the developer might not run the entire test suite locally (very common for large test suites).

To fix this, we can use `unittest.mock` to achieve the same effect.

```python
from unittest import mock

def test_we_are_back_in_the_90s():
    with mock.patch("datetime.date", NewDate):
        service = MyService()
        result = service.get_current_time()
        assert result.year == 1990
```

Or equivalently we can use the `pytest` `monkeypatch` function-scoped fixture to properly clean up the monkey-patching after the test is run.

```python
def test_we_are_back_in_the_90s(monkeypatch):
    monkeypatch.setattr("datetime.date", NewDate)
    service = MyService()
    result = service.get_current_time()
    assert result.year == 1990
```

**Tips to detect this kind of flakiness** To help detect state pollution issues, you can use the [pytest-randomly](https://pypi.org/project/pytest-randomly/) plugin to run your tests in random order. This way, you can increase the likelihood of catching state pollution issues if you run your test suite multiple times with different random seeds. Worth pointing out another similar plugin [pytest-random-order](https://pypi.org/project/pytest-random-order/).

Furthermore, using plugins that run your test suite in parallel like [pytest-xdist](https://pypi.org/project/pytest-xdist/) can help you catch state pollution issues. This is because running your test suite in parallel will not adhere to the order of your tests and will increase the likelihood of catching state pollution issues.

#### Database state pollution

Improper isolation of database state can lead to inter-test flakiness. Consider the following:

```python
# ----------------------------------
# Implementation - create_user.py
# ----------------------------------
import pytest

class CreateUserAction:
    def __init__(self, name):
        self.name = name

    def run(self, db):
        sql = (
            "INSERT INTO test_users_table (name) "
            f"VALUES ('{self.name}');"
        )
        with db.conn as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

# ----------------------------------
# test suite - conftest.py
# ----------------------------------
import os
import psycopg2

class TestDatabase:
    def __init__(self, db_url=None):
        if db_url is None:
            db_url = os.environ["TEST_DATABASE__URL"]
        self.conn = psycopg2.connect(db_url)

    def setup(self):
        sql = (
            "CREATE TABLE IF NOT EXISTS "
            "test_users_table (name VARCHAR(255));"
        )
        with self.conn as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def teardown(self):
        self.conn.close()


@pytest.fixture(scope="session")
def db():
    db = TestDatabase()
    db.setup()
    yield db
    db.teardown()


# ----------------------------------
# test suite - test_create_user.py
# ----------------------------------

def test_create_user_action(db):
    # count number of users before adding a user
    count_sql = (
        "SELECT COUNT(*) FROM test_users_table;"
    )
    with db.conn as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql)
            count_before_adding_users = (
                cur.fetchone()[0]
            )

    # add a user
    action = CreateUserAction(name="Alice")
    action.run(db)

    # count number of users after adding a user
    with db.conn as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql)
            count_after_adding_users = (
                cur.fetchone()[0]
            )

    # check count only incremented by 1 - right?
    assert (
        count_after_adding_users == 
        count_before_adding_users + 1
    )
```

This test is flaky because it does not properly isolate the database state. 

More specifically, let's think of what happens when more than one process is running the same test function which is a common scenario in CI pipelines where multiple workflows are running in parallel.

Given that the database is not properly isolated, the following sequence of events could occur:
- Process 1 runs on machine 1 and executes the test `test_create_user_action` - it triggers the creation of a user named Alice
- Process 2 runs on machine 2 almost at the same time and it executes the test `test_create_user_action` - it triggers the creation of a user named Alice right before Process 1 checks the number of users in the database
- Process 1 checks the number of users in the database and finds that `count_after_adding_users == count_before_adding_users + 1` is False and the test fails

Now you will have to be awfully unlucky for this to happen. However, it is not impossible and becomes more likely as the test suite grows in size and as the number of processes running the test suite increases.

To fix this, we can resort to using temporary tables to isolate the database state. This way, each test session will have its own temporary testing table to work with and the database state will be properly isolated.

```python
...
# ----------------------------------
# test suite - conftest.py
# ----------------------------------
import os
import psycopg2

class TestDatabase:
    def __init__(self, db_url=None):
        if db_url is None:
            db_url = os.environ["TEST_DATABASE__URL"]
        self.conn = psycopg2.connect(db_url)

    def setup(self):
        sql = (
            "CREATE TEMPORARY TABLE IF NOT EXISTS "
            "test_users_table (name VARCHAR(255));"
        )
        with self.conn as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def teardown(self):
        self.conn.close()

@pytest.fixture(scope="function")
def db():
    db = TestDatabase()
    db.setup()
    yield db
    db.teardown()
```

A temporary table is a table that is automatically dropped at the end of a session. This means that each database connection will have its own temporary table to work with and the database state will be properly isolated. 

Note that I also changed the fixture scope to function so if we have multiple tests modifying the `test_users_table` they will not interfere with each other when run in parallel.


**Tip on detecting inter-test flakiness** When thinking of inter-test flakiness, it is worth considering the following questions:
- Does one test modify a shared state that another test relies on?
- Does one test modify a shared state such that a parallel run of the same test will fail?


## External Factors

External factors of flakiness are factors that are outside of the control of the test suite. They are factors that are not related to the implementation of the test suite nor the implementation of the code under test.

### Network issues

Network issues can lead to flakiness. For example, if your test suite is making requests to an external API, it is possible that the API is down or that the network is down. This can lead to flakiness.

```python
# ----------------------------------
# implementation - query_example.py
# ----------------------------------
import requests

def query_web_server():
    response = requests.get("https://example.com")
    return response.status_code

# ----------------------------------
# test suite - test_query_example.py
# ----------------------------------
def test_query_web_server():
    code = query_web_server()
    assert (
        code == 200, 
        f"Expected status code 200, but got {code}"
    )
```

In general, it is a good idea to avoid making network calls in your test suite. However, if you must, you can resort to using a mocking library like [responses](https://pypi.org/project/responses/) to mock the network calls.

**Tip on preventing network-related issues** It is worth noting that there is a pytest plugin called [pytest-socket](https://pypi.org/project/pytest-socket/) that can be used to disable socket calls during tests.

### Asynchronous waiting on external resources

Another common source of flakiness is asynchronous waiting on external resources. Consider the following example where we wait for a file to be updated before we read it. The external system that updates the file is not under our control and it is possible that it is slow to update the file. We mimic this behavior by introducing a random delay in the test.

```python
# ----------------------------------
# implementation - record_keeper.py
# ----------------------------------
import asyncio
import aiofiles
import random

class RecordKeeper:
    def __init__(self, filepath):
        self.filepath = filepath

    async def update_file(self, contents):
        # mimic slow update
        await asyncio.sleep(random.randint(1, 3))
        
        async with aiofiles.open(self.filepath, "w") as f:
            print("writing to file")
            await f.write(contents)
            print("done writing to file")

# ----------------------------------
# test suite - test_record_keeper.py
# ----------------------------------
import pytest

@pytest.mark.asyncio
async def test_record_keeper(tmpdir):
    filepath = tmpdir / "test.txt"
    keeper = RecordKeeper(filepath)
    # update is performed in the background
    asyncio.create_task(keeper.update_file("hello world"))

    # wait for a fixed amount of time
    await asyncio.sleep(2)

    async with aiofiles.open(filepath, "r") as f:
        contents = await f.read()
        assert contents == "hello world"
```

While this example might seem contrived, it is not uncommon to have to wait for external resources to be updated before we can read them. This is especially true when dealing with testing front-end applications where we might have to wait for the DOM to be updated before we can assert on it.

**Tip on avoiding async wait flakiness** To avoid this kind of flakiness, avoid making use of heuristics to wait for external resources to be updated. Instead, resort to using explicit await statements to wait for the external resource to be updated. Additionally if its a unit test, you can mock the external resource to be updated immediately.


## Wrap up

I hope the above examples will help you spot flakiness in your test suite and move you one step closer to a more reliable test suite. Having confidence in a test suite is important and will help you and your team rest easy. 

To avoid this article becoming too long, I did not cover all the possible sources of flakiness. I encourage you to read the [Surveying the developer experience of flaky tests](https://www.gregorykapfhammer.com/download/research/papers/key/Parry2022-paper.pdf) paper for a comprehensive overview. Additionally you can refer to the [flakycode](https://www.github.com/marwan116/flakycode) github repository for the complete code examples.