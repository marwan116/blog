---
title: Compressing contiguous ranges
date: 2022/11/28
description: A simple approach to compress contiguous ranges.
tag: algorithms, data structures, python, date wrangling
author: Marwan
---

# Compressing contiguous ranges

In this post, we will discuss a simple approach to compress contiguous ranges.

The problem can be described as follows:

Take a list of values like this:

```python
dates = [
    date(2022, 10, 1),
    date(2022, 10, 2),
    date(2022, 10, 3),
    date(2022, 10, 6),
    date(2022, 10, 8)
]
```

and compress the values into a list of tuples like such that the first value in the tuple is the start of the range and the second value is the end of the contiguous range (A contiguous range is defined as a range of values where the difference between each value is 1).

For example, the above list of values would be compressed into the following list of tuples:

```python
ranges = [
    (date(2022, 10, 1), date(2022, 10, 3)),
    (date(2022, 10, 6), date(2022, 10, 6)),
    (date(2022, 10, 8), date(2022, 10, 8))
]
```

Note that the values can be any type where a difference can be calculated (i.e. supports __sub__) (e.g. `int`, `date`, `datetime`, `pd.Timestamp`, `pd.Period`, ...).  

This problem can also take the form of compressing any iterable, even a pandas Series.

For example, the following input:

```python
import pandas as pd

periods = pd.Series([
    pd.Period("2022-10-1", freq="D"),
    pd.Period("2022-10-2", freq="D"),
    pd.Period("2022-10-3", freq="D"),
    pd.Period("2022-10-6", freq="D"),
    pd.Period("2022-10-8", freq="D")
])
```

would then be converted into a either a series of tuples, or a pandas DataFrame:

```python
ranges = pd.Series([
    (pd.Period("2022-10-1", freq="D"), pd.Period("2022-10-3", freq="D")),
    (pd.Period("2022-10-6", freq="D"), pd.Period("2022-10-6", freq="D")),
    (pd.Period("2022-10-8", freq="D"), pd.Period("2022-10-8", freq="D"))
])
```

or

```python
ranges = pd.DataFrame({
    "start": [
        pd.Period("2022-10-1", freq="D"),
        pd.Period("2022-10-6", freq="D"),
        pd.Period("2022-10-8", freq="D")
    ],
    "end": [
        pd.Period("2022-10-3", freq="D"),
        pd.Period("2022-10-6", freq="D"),
        pd.Period("2022-10-8", freq="D")
    ]
})
```

This problem is also commonly referred to as "gaps and islands" or "gaps and islands problem".

# Solution

## A pandas based solution is shown below:

```python
In [1]: import pandas as pd
   ...: 
   ...: values = pd.Series([
   ...:     pd.Period("2022-10-1", freq="D"),
   ...:     pd.Period("2022-10-2", freq="D"),
   ...:     pd.Period("2022-10-3", freq="D"),
   ...:     pd.Period("2022-10-6", freq="D"),
   ...:     pd.Period("2022-10-8", freq="D")
   ...: ])
   ...: 
   ...: # Sort the values
   ...: values = values.sort_values()
   ...: 

In [2]: 
   ...: # Calculate the difference between each value
   ...: diff = pd.to_timedelta(values.diff()).dt.days
   ...: 

In [3]: diff
Out[3]: 
0    NaN
1    1.0
2    1.0
3    3.0
4    2.0
dtype: float64

In [4]: # Mark the start of each range
   ...: start = diff != 1
   ...: 
   ...: start
Out[4]: 
0    True
1    False
2    False
3     True
4     True
dtype: bool

In [5]: start.cumsum()
Out[5]: 
0    0
1    0
2    0
3    1
4    2
dtype: int64

In [6]: # Group the values by the start of each range
   ...: groups = values.groupby(start.cumsum())
   ...: 

In [7]: groups
Out[7]: <pandas.core.groupby.generic.SeriesGroupBy object at 0x11e533f10>

In [8]: # Create a DataFrame of the start and end of each range
   ...: ranges = pd.DataFrame({
   ...:     "start": groups.first(),
   ...:     "end": groups.last()
   ...: })

In [9]: ranges
Out[9]: 
        start         end
0  2022-10-01  2022-10-03
1  2022-10-06  2022-10-06
2  2022-10-08  2022-10-08
```


In the above solution, we:
- sort the values
- calculate the difference between each value.
- mark the start of each range by checking if the difference between each value is greater than 1.
- group the values by the start of each range
- create a DataFrame of the start and end of each range.

The unique part of this solution is the use of `start.cumsum()` to group the values by the start of each range.  

One thing to point out is the comparison operation (diff != 1) evaluates to True for the first value given equal comparisons against np.Nan always evaluate to False. 


## A plain python solution is shown below:

```python
In [1]: from datetime import date, timedelta
   ...: 
   ...: values = [
   ...:     date(2022, 10, 1),
   ...:     date(2022, 10, 2),
   ...:     date(2022, 10, 3),
   ...:     date(2022, 10, 6),
   ...:     date(2022, 10, 8)
   ...: ]
   ...:

In [2]: # Sort the values  
    ...: values = sorted(values)
    ...:

In [3]: # Calculate the difference between each value
    ...: diff = [b - a for a, b in zip(values, values[1:])]
    ...:

In [4]: diff
Out[4]: [
 timedelta(days=1),
 timedelta(days=1),
 timedelta(days=3),
 timedelta(days=2)]

In [5]: # Mark the start of each range
    ...: start = [True] + [d.days != 1 for d in diff]
    ...:

In [7]: start
Out[7]: [True, False, False, True, True]

In [8]: from itertools import accumulate, groupby
    ...: 
    ...: cumsum = map(int, accumulate(start))
    ...: groups = groupby(values, key=lambda x: next(cumsum))
    ...: values = [list(v) for k, v in groups]
    ...: ranges = [(v[0], v[-1]) for v in values]

In [9]: ranges
Out[9]: 
[(datetime.date(2022, 10, 1), datetime.date(2022, 10, 3)),
 (datetime.date(2022, 10, 6), datetime.date(2022, 10, 6)),
 (datetime.date(2022, 10, 8), datetime.date(2022, 10, 8))]
```

The plain python solution is more verbose but follows the same logic as the pandas solution.  The main difference is the use of itertools.accumulate and itertools.groupby to compute cumulative sums and group the values by the start of each range.

The python implementations of accumulate and groupby are shown on the [itertools documentation page here](https://docs.python.org/3/library/itertools.html#itertools.groupby)

