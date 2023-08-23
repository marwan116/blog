---
title: "A showcase of Flaky Tests: How not to footgun yourself when writing tests"
date: 2023/08/15
description: Valuable learnings about flaky tests inspired by a talk python podcast titled "Taming Flaky Tests"
tag: testing, python, pytest
author: Marwan
---

# Intro
This article is primarily inspired by a [talk python](https://talkpython.fm/) podcast episode titled "Taming Flaky Tests" where Michael Kennedy, the host of the podcast, interviewed [Gregory M. Kapfhammer](https://www.gregorykapfhammer.com/), an Associate Professor in the Department of Computer Science at Allegheny College, and [Owain Parry](https://www.linkedin.com/in/owain-parry-a0040a216), a Computer Science PhD student at the University of Sheffield whose research focuses on software testing. 

To avoid this article from being a simple regurgitation of the transcript of the podcast episode, I focus on providing code examples (hopefully not too contrived) to illustrate some of the main concepts mentioned in the episode or in the show notes. 

# Table of Contents

This article is mainly a showcase of code examples of flaky tests. I have borrowed a categorization of the flaky tests referenced in the paper [Surveying the developer experience of flaky tests](https://www.gregorykapfhammer.com/download/research/papers/key/Parry2022-paper.pdf) by Owain Parry, Gregory M. Kapfhammer, Michael Hinton and Phil McMinn.

Accordingly, here is a table of contents of the different types of flaky tests that I will be covering in this article:

- Intra-test flakiness
    - Concurrency - The GIL won't save you
    - Randomness
        - Algorithmic non-determinism
    - Floating point arithmetic
    - Missing corner cases (test is too restrictive)
    - Timeout - Make sure your timeout is not too short
- Inter-test flakiness
    - Test order dependency
        - State pollution by incorrect mocking/monkey-patching
        - Incorrect scoping of fixtures
        - Database state pollution
- External factors
    - Network
    - File system


## Intra-Test Flakiness 
Intra-test generally means flakiness stemming from a single test and not due to the interference of other tests or due to the interference of external factors like the network or the file system.

### Concurrency - The GIL won't save you

First up is an example of a test that is attempting to make use of concurrency to speed up the test run time only to shoot itself in the foot by introducing flakiness.

Can you spot the problem with this test example? Hint: always think twice before reaching out to the threading module to speed up your test suite.

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
    def check_account_by_charging_and_refunding_small_amounts(
        self, account, test_fee=0.01, num_checks=400
    ):
        for _ in range(num_checks):
            account.withdraw(test_fee)
            account.deposit(test_fee)

# ----------------------------------
# test suite - test_bank_account.py
# ----------------------------------
import threading

def test_charge_and_refund_keeps_the_balance_the_same():
    # initialize bank account with $100 balance
    account = BankAccount()
    original_balance = account.balance

    threads = []

    # "smartly" parallelize the call to check_account_by_charging_and_refunding_small_amounts
    merchants = [Merchant() for _ in range(10)]
    for merchant in merchants:
        thread = threading.Thread(
            target=merchant.check_account_by_charging_and_refunding_small_amounts,
            args=(account,),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert account.balance == original_balance
```

This test is flaky because the `BankAccount` implementation is not thread-safe. This is because access to the `balance` attribute is not protected by a lock meaning that multiple threads can concurrently access and modify the balance attribute.

More specifically, in the above test, multiple threads concurrently call the `check_account_by_charging_and_refunding_small_amounts` function, which performs a series of BankAccount `withdraw` and BankAccount `deposit` operations.

Threads may interleave in a way that one thread runs `deposit` while another thread runs `withdraw` at the same time.

The GIL contrary to "popular fallacy" does not provide atomicity or synchronization guarantees for complex operations involving multiple bytecode instructions. 

To ensure thread safety in scenarios like this, you would need to use proper synchronization mechanisms like locks to protect critical sections of code, ensuring that only one thread can modify shared data at a time.

Pro tip, if you want to make it more likely that your test suite will catch this concurrency issue, it is recommended you:
- set a smaller switch interval to force the interpreter to switch between threads more frequently
- run your test a reasonably large number of times 

i.e. see the example below:

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
        print("passed_percent", passed / n_iter * 100, "%")
    finally:
        sys.setswitchinterval(original_switch_interval)
```

Note that the switch interval is a global setting and as such it will affect the entire test suite if you modify it before running more tests. As such, it is recommended that you reset the switch interval to its original value to check your specific test.

For more details on the switch interval, see this article from [superfastpython](https://superfastpython.com/context-switch-interval-in-python/#Why_Change_the_Switch_Interval)

### Randomness

Introducing randomness into test inputs offers some benefits. It can help you test a wider range of inputs and it can help you avoid biasing your tests towards a specific input. However, it can also lead to flakiness if it is not properly handled.

#### Algorithmic non-determinism

Let's start with an example of what could be considered algorithmic non-determinism. 

Consider the following:

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
    return minimize(rosenbrock, initial_guess, method='Nelder-Mead')

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

    assert np.all(np.isclose(result.x, [1, 1], atol=naively_chosen_atol))
```

This test is flaky because the chosen `Nelder-Mead` minimization algorithm is not always guaranteed to converge to the same true minimum - i.e. the algorithm is not deterministic.

More specifically, in this example we are using `Nelder-Mead` to minimize the `rosenbrock` function, also referred to as the Valley or Banana function, which is a popular test problem for minimization algorithms. The function is unimodal, and the global minimum lies in a narrow, parabolic valley. The narrow valley makes it somewhat difficult for the algorithms to converge exactly to the true minimum (1, 1) given different starting points.

In this case, the tester, knowing that the `Nelder-Mead` will only offer an approximate value of the true minimum, chose a "naive" tolerance value of 1e-5 without proper consideration to the expected variation in the result of the optimization algorithm (the hand-waving here might be that well 1e-5 is a relatively large tolerance value compared to what I tried before and it makes the test pass when I run it on my machine).

Turns out the chosen tolerance value of 1e-5 makes the test flaky fail almost 65% of the time given the initial guess is randomly chosen from the range 0 to 10. Important to note here that while too small a tolerance value leads to false alarms, naively solving this issue by updating to a very large tolerance value means that the test is not sensitive enough to detect when the result is not within a statistically reasonable range.

To remedy this, we assess the variance of the results of running the `Nelder-Mead` algorithm multiple times. Then we chose the tolerance to be 3 standard deviations from the mean. This way, we can be 99.7% confident (assuming the results are normally distributed) that the test will pass (not too permissive, not too strict). 

The updated test looks like this:

```python
....

def estimate_tolerance(num_runs=50):
    results = []

    for _ in range(num_runs):
        initial_guess = np.random.randint(0, 10, size=2)
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

    # tolerance is estimated from results of running minimization multiple times
    tolerance = estimate_tolerance()

    assert np.all(np.isclose(result.x, [1, 1], atol=naively_chosen_atol))
```

### Floating point arithmetic

Dealing with floating point arithmetic can be tricky. For instance, it is a common practice to resort to using 32-bit data types like float32 and int32 to save memory especially when dealing with large arrays of data (i.e. in pandas or numpy). However, this can be the source of test flakiness if you are not careful.

Let's start with a tricky question inspired by the lovely article ["The problem with float32: you only get 16 million values" on pythonspeed.com](https://pythonspeed.com/articles/float64-float32-precision/)

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
The outcome is now False - because in laymen terms beyond 2^18 (262,144) the float32 data type cannot maintain a precision of 1/64 (0.015625).

#### Underflow or overflow issues

Consider the following test showcasing an overflow issue when using np.int32:

```python
import pytest
import numpy as np

def compute_eng_balance(all_expenses, is_eng_dept):
    total_eng_budget = np.int32(0)
    for balance, is_eng in zip(all_expenses, is_eng_dept):
        if is_eng:
            total_eng_budget += balance
    return total_eng_budget

def test_eng_balance_is_correctly_computed():
    eng_dept_cost = 1_600_000_000
    non_eng_dept_cost = 1_400_000_000

    num_dept = 10
    num_eng_dept = np.random.randint(1, num_dept)
    num_non_eng_dept = num_dept - num_eng_dept
    is_eng_dept = np.array(
        [True] * num_eng_dept + [False] * num_non_eng_dept, dtype="bool"
    )
    all_expenses = np.array(
        [eng_dept_cost] * num_eng_dept + [non_eng_dept_cost] * num_non_eng_dept, dtype=np.int32
    )
    total_eng_budget = compute_eng_balance(all_expenses, is_eng_dept)
    assert np.isclose(total_eng_budget, eng_dept_cost * num_eng_dept)
```

We check if the total engineering budget computed by summing up the balances of all engineering departments is equal to the total engineering budget computed by multiplying the engineering department cost by the number of engineering departments.

This test is flaky due to an overflow in np.int32 - more specifically the maximum value for np.int32 is 2,147,483,647 - which can be verified by running the following code in a python interpreter:

```python
In [1]: import numpy as np

In [2]: np.iinfo(np.int32).max
Out[2]: 2147483647
```

Given 1 engineering department with a cost of 1.6 billion still smaller than the max value of np.int32 ~2.1 billion, the test passes. However, given 2 engineering departments with a cost of 1.6 billion, the test fails due to an overflow in np.int32.

This can be fixed by making our implementation more robust to overflow either by using a larger data type like np.int64 or by forcing an overflow error to be raised. 

The latter can be achieved by using np.seterr to set the overflow error to be raised. 

```python
def compute_eng_balance(all_expenses, is_eng_dept):
    np.seterr(over="raise")
    ...
```

In general, when dealing with billions, one has to make the conscious decision to use a larger data type like np.int64 or np.float64.

#### Loss in precision

One might think that switching to a floating point data type like np.float32 would solve the problem of overflow without having to pay the cost of using a larger data type like np.int64. However, a different problem arises when using floating point data types - loss in precision.

```python
def test_balance_zeros_out():
    eng_dues = np.random.choice(
        [1_600_000_000, 1_630_000_000]
    )
    num_dept = 10
    usd_balance = np.array([eng_dues] * 10, dtype=np.float32)

    num_eng = np.random.randint(1, num_dept)
    num_non_eng = num_dept - num_eng
    is_eng_dept = np.array([True] * num_eng + [False] * num_non_eng, dtype="bool")

    remaining_eng_spend = np.float32(eng_dues * num_eng)
    for balance, is_eng in zip(usd_balance, is_eng_dept):
        if is_eng:
            remaining_eng_spend -= balance

    assert np.isclose(remaining_eng_spend, 0)
```

In this example, we check if the remaining engineering spend is equal to 0. We will define a `precise_float32` to help us illustrate the problem.

```python
In [1]: import numpy as np

In [2]: 
    ...: def precise_float32(value):
    ...:     tol = np.finfo(np.float32).eps
    ...:     if np.abs(np.float32(value) - np.float64(value)) > tol:
    ...:         raise ValueError("Loss of precision")
    ...:     return np.float32(value)
    ...: 

In [21]: precise_float32(value=1.63e9 * 2)
Out[21]: 3260000000.0

In [4]: precise_float32(value=1.63e9 * 3)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[4], line 1
----> 1 precise_float32(value=1.63e9 * 3)

Cell In[17], line 4, in precise_float32(value)
      2 tol = np.finfo(np.float32).eps
      3 if np.abs(np.float32(value) - np.float64(value)) > tol:
----> 4     raise ValueError("Loss of precision")
      5 return np.float32(value)

ValueError: Loss of precision
```

Other approachs to make the implementation more reliable is to resort to division and subtraction to transform large number operations into smaller number operations to avoid loss of precision or overflow issues.

### Missing corner cases (test is too restrictive)

Sometimes, a test is too restrictive and does not account for all possible corner cases. This can lead to flakiness.

Consider the following:

```python
import pytest
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, column


class FrameStacker:
    def stack(self, data):
        return data.stack()

    def unstack(self, data):
        return data.unstack()


@pytest.fixture
def data():
    """A way to mimick randomly generated data."""
    nrows = np.random.randint(0, 10)
    return pd.DataFrame({
        "A": np.random.choices([1.0, 2.0, 3.0, np.nan], k=nrows),
        "B": np.random.choices([1.0, 2.0, 3.0, np.nan], k=nrows),
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

One way to find the above-mentioned corner cases is to use fuzzing. Fuzzing is a technique where you 
run a "testing campaign" that generates multiple random inputs against your code to find corner cases. 

One common approach to fuzzing is to use a property-based testing framework like [hypothesis](https://hypothesis.readthedocs.io/en/latest/).

Here is an example of how you can use hypothesis to find the corner cases mentioned above:

```python
@given(data=data_frames(
    columns=[
        column(name="A", dtype=float, elements=st.floats(allow_nan=True)),
        column(name="B", dtype=float, elements=st.floats(allow_nan=True)),
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

Adding timeouts to tests can be a good way to ensure that tests do not take too long to run. However, if the timeout is too short, it can lead to flakiness.

```python
import numpy as np
from scipy.optimize import newton

# ----------------------------------
# implementation - find_discount_rate.py
# ----------------------------------
def calculate_present_value(discount_rate, cashflows):
    return np.sum(cashflows / (1 + discount_rate) ** np.arange(1, len(cashflows) + 1))


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
@pytest.mark.timeout(2)
def test_find_discount_rate():
    cashflows = np.random.randint(50, 300, size=360_000)
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

Mocking and monkey-patching are powerful tools that can be used to isolate tests from external dependencies. However, if not used properly, they can lead to inter-test flakiness.

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
    datetime.date = NewDate
    service = MyService()
    result = service.get_current_time()
    assert result.year == 1990

def test_we_are_in_the_21st_century():
    assert datetime.date.today().year >= 2000
```

In this example, we have a service that returns the current time. We have a test that
checks if the current time is in the 90s. We also have another test that checks if the
current time is in the 21st century.

The test `test_we_are_back_in_the_90s` introduces the flakiness because it incorrectly monkey-patches `datetime.date` to return January 1st 1990. This monkey-patching is not properly cleaned up after the test is run. 

As such, the test `test_we_are_in_the_21st_century` will fail because it will be run after the first test and it will be run in the 90s.

How is this flaky you might ask - if a developer runs this on their machine shouldn't they immediately get an error and fix it? Well, not necessarily.

- The two tests might be in separate files and the developer might have not run the entire test suite locally (understandable for large test suites).
- The CI pipeline might be using a plugin like `pytest-randomly` to introduce randomness in the order of tests or a plugin like `pytest-xdist` to parallelize the test runs. This means that the tests might not be run in the same order they are parsed by the test runner.

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



#### Incorrect scoping of fixtures

In some cases, developers might chose to provide wider-scoped fixtures to speed up the test suite. However, this can lead to inter-test flakiness.

This is an almost contrived example of how a fixture that is scoped to the module can lead to flakiness.

```python
import pytest

class TicTacToeSimulation:
    def __init__(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.moves_made = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.moves_made < 9:
            self.moves_made += 1
            return self
        raise StopIteration

    def make_move(self, row, col):
        if self.board[row][col] == " ":
            self.board[row][col] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"

    def get_board(self):
        return self.board

@pytest.fixture(scope="module")
def tic_tac_toe_simulation():
    return TicTacToeSimulation()

def test_player_moves(tic_tac_toe_simulation):
    for _ in tic_tac_toe_simulation:
        tic_tac_toe_simulation.make_move(0, 0)
    board = tic_tac_toe_simulation.get_board()
    assert board == [["X", " ", " "], [" ", " ", " "], [" ", " ", " "]]

def test_winning_move(tic_tac_toe_simulation):
    for _ in tic_tac_toe_simulation:
        tic_tac_toe_simulation.make_move(0, 0)
        tic_tac_toe_simulation.make_move(0, 1)
        tic_tac_toe_simulation.make_move(1, 0)
        tic_tac_toe_simulation.make_move(1, 1)
        tic_tac_toe_simulation.make_move(2, 0)
    board = tic_tac_toe_simulation.get_board()
    assert board == [["X", "X", " "], ["O", "O", " "], ["X", " ", " "]]
```

In this example, we have a test that checks if the first value of an iterator is 1. We also have another test that checks if the iterator consumes values.

The first test is flaky because the iterator fixture is scoped to the module. This means that the iterator is shared between tests. 

#### Database state pollution

Improper isolation of database state can lead to inter-test flakiness. Consider the following:

```python
import pytest
import psycopg2


class TestDatabase:
    def __init__(self, host="localhost", port=5432, name="postgres"):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=name,
        )

    def setup(self):
        with self.conn as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS test_users_table (name VARCHAR(255));"
                )

    def teardown(self):
        self.conn.close()


class CreateUserAction:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, db: TestDatabase):
        with db.conn as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO test_users_table (name) VALUES ('{self.name}');"
                )


@pytest.fixture(scope="module")
def db():
    db = TestDatabase()
    db.setup()
    yield db
    db.teardown()


def test_create_user_action(db):
    with db.conn as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM test_users_table;")
            count_before_adding_users = cur.fetchone()[0]

    CreateUserAction(name="Alice")(db)

    with db.conn as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM test_users_table;")
            count_after_adding_users = cur.fetchone()[0]

    assert count_after_adding_users == count_before_adding_users + 1
```

### In software engineering, everything is a trade-off

When it comes to software testing, there is no silver bullet. There are always trade-offs to be made. For example, when writing tests, you can choose to write tests that are fast or tests that are reliable. You can't have both. Reliable tests necessitate more developer time and longer test-run times. Developer productivity and efficiency of testing process.

Tricks like using markers to mark slow tests to avoid hit to developer productivity could ameliorate this.

### How to approach flaky tests

Ignoring flaky tests or forcing re-runs of flaky tests until they pass is not always the best approach. Flaky tests could be pinpointing a weak spot in your code. They could be a symptom of a deeper problem. It is important to investigate the root cause of flaky tests and fix them.

As such there is a silver lining to flaky tests. They can be used as a tool to improve the quality of your code and they can force you to think about the design of your code and the design of your tests. (e.g. if you have a flaky test due to unanticipated concurrency issues, you might want to think about how to make your code more thread-safe)

### Finding flaky tests
- Resorting to fuzzing by using a property-based testing framework like [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) which generates random inputs to test your code
- Using [pytest-randomly]() to force the running of tests in random order
- Running a test-suite in parallel using [pytest-xdist]() to root out tests that are not properly designed to run in parallel


### Learnings from big-tech companies

Both google and spotify have written about their experiences with flaky tests and how they have dealt with them.

- [Google](https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html)
- [Spotify](https://engineering.atspotify.com/2019/11/test-flakiness-methods-for-identifying-and-dealing-with-flaky-tests/)

For Google, they have a dedicated team of engineers that are responsible for monitoring and fixing flaky tests. They have a dashboard that tracks the flakiness of tests and they have a process for dealing with flaky tests. They have a policy that if a test fails 3 times in a row, it is automatically disabled and the team that owns the test is notified. The team then has 2 days to fix the test. If the test is not fixed within 2 days, it is deleted. They also have a policy that if a test fails 3 times in a row, the code that it is testing is automatically reverted. This is to prevent flaky tests from blocking the release of new code.

Quarantining flaky tests is one approach to take. As long as developers are serious about revisiting the quaranteened tests - i.e. as long as it doesn't become one big trash can.

Psychological impact of flaky tests - sort of like boy who cried wolf. If you have a flaky test that fails 3 times in a row, you might be tempted to ignore it the 4th time it fails. This is a dangerous mindset to have. You should always investigate the root cause of flaky tests and fix them.

Trust and faith in a test-suite is important. If you have a flaky test-suite, all of a sudden it will degrade the trust and faith that developers have in the test-suite. They will be less likely to run the test-suite and they will be less likely to fix failing tests. This is a dangerous situation to be in. You want to have a test-suite that developers trust and have faith in.
                                                                        