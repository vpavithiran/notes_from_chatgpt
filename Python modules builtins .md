# Python builtins guide

## Prompt:
act as an expert in "python module expert" focused on teaching me your task is to teach me complete python module expert basic to advance complete. 
i will give you each module name in each prompt...you must give full content about the module...in depth with explaination and try to give all the content in the give module

## Response:
Absolutely, Pavithiran! I can do that. Send me the name of the Python module you want to start with, and I’ll provide a **complete, in-depth guide** covering:

*   Overview and purpose of the module
*   Installation and setup
*   All important classes, functions, and methods
*   Parameters, options, and return values
*   Practical usage examples
*   Best practices, tips, and advanced usage

Once you give me a module name, I’ll create a **full, structured tutorial** for it from **basic to advanced**.

Which module do you want to start with?

## Prompt:
builtins

## Response:
Perfect! Let’s dive deep into Python’s **`builtins`** module. This is a foundational module in Python, as it contains all the standard objects that are automatically available without importing anything. Understanding it thoroughly is essential for becoming a Python expert.

* * *

**Python `builtins` Module – Complete Guide**
=============================================

**1\. Overview**
----------------

*   The `builtins` module contains **all the built-in objects** in Python.
*   These are available **automatically** in every Python environment, so you don’t need to import them to use them.
*   Examples of built-ins include:
    *   Data types: `int`, `str`, `list`, `dict`, etc.
    *   Functions: `print()`, `len()`, `sum()`, etc.
    *   Exceptions: `ValueError`, `TypeError`, `ZeroDivisionError`, etc.
    *   Constants: `True`, `False`, `None`.

> **Key point:** `builtins` can also be explicitly imported using `import builtins`, which allows you to inspect or override built-in functions safely.

```python
import builtins
print(dir(builtins))  # Lists all built-in functions, exceptions, and types
```

* * *

**2\. Accessing Built-in Functions**
------------------------------------

Python has **many built-in functions**. Here’s a categorization with examples:

### **2.1 Type Conversion Functions**

*   `int(x)` → converts to integer
*   `float(x)` → converts to float
*   `str(x)` → converts to string
*   `bool(x)` → converts to boolean
*   `complex(real, imag)` → converts to complex number

```python
print(int("10"))        # 10
print(float("3.14"))    # 3.14
print(str(100))         # '100'
print(bool(0))          # False
print(complex(2, 3))    # (2+3j)
```

* * *

### **2.2 Sequence and Collection Functions**

*   `len(obj)` → length of iterable
*   `sum(iterable)` → sum of numbers
*   `max(iterable)` → maximum value
*   `min(iterable)` → minimum value
*   `sorted(iterable)` → returns sorted list
*   `reversed(seq)` → returns reverse iterator

```python
numbers = [4, 2, 9]
print(len(numbers))    # 3
print(sum(numbers))    # 15
print(max(numbers))    # 9
print(min(numbers))    # 2
print(sorted(numbers)) # [2, 4, 9]
print(list(reversed(numbers)))  # [9, 2, 4]
```

* * *

### **2.3 Object Inspection Functions**

*   `type(obj)` → returns the type of an object
*   `isinstance(obj, class)` → checks if object is instance of class
*   `issubclass(subclass, class)` → checks inheritance
*   `id(obj)` → memory address of object
*   `dir(obj)` → list of attributes and methods

```python
x = [1, 2, 3]
print(type(x))             # <class 'list'>
print(isinstance(x, list)) # True
print(issubclass(bool, int))  # True
print(id(x))               # Memory address
print(dir(x))              # ['append', 'clear', 'copy', ...]
```

* * *

### **2.4 Mathematical Functions**

*   `abs(x)` → absolute value
*   `round(x, n)` → rounds to n decimals
*   `divmod(a, b)` → returns (quotient, remainder)
*   `pow(x, y)` → x^y

```python
print(abs(-5))       # 5
print(round(3.14159, 2))  # 3.14
print(divmod(17, 3)) # (5, 2)
print(pow(2, 3))     # 8
```

* * *

### **2.5 Input and Output**

*   `print(*objects)` → prints to console
*   `input(prompt)` → reads input from user

```python
name = input("Enter your name: ")
print(f"Hello, {name}!")
```

* * *

### **2.6 Iterables and Functional Programming Helpers**

*   `enumerate(iterable)` → index + element pairs
*   `zip(*iterables)` → combine iterables elementwise
*   `filter(func, iterable)` → filter elements
*   `map(func, iterable)` → apply function to each element
*   `any(iterable)` → True if any element is True
*   `all(iterable)` → True if all elements are True

```python
nums = [1, 2, 3, 4]
print(list(enumerate(nums)))        # [(0,1),(1,2),(2,3),(3,4)]
print(list(zip([1,2], ['a','b']))) # [(1,'a'),(2,'b')]
print(list(filter(lambda x: x>2, nums))) # [3,4]
print(list(map(lambda x: x**2, nums)))   # [1,4,9,16]
print(any([0,0,1]))   # True
print(all([1,2,3]))   # True
```

* * *

### **2.7 Object Management**

*   `del obj` → delete object
*   `globals()` → dictionary of global variables
*   `locals()` → dictionary of local variables
*   `callable(obj)` → checks if object is callable
*   `hasattr(obj, 'attr')` → check attribute
*   `getattr(obj, 'attr')` → get attribute
*   `setattr(obj, 'attr', value)` → set attribute
*   `hash(obj)` → hash value of object

```python
x = 10
print(globals()['x'])  # 10
print(callable(print)) # True
print(hasattr(x, 'bit_length')) # True
print(getattr(x, 'bit_length')()) # 4
setattr(x, 'new_attr', 5)  # Fails for int, works for objects
```

* * *

### **2.8 Exception Classes**

Python built-ins also include **all exceptions**:

*   `Exception` → base class
*   `ValueError`, `TypeError`, `ZeroDivisionError`, `IndexError`, `KeyError`, etc.

```python
try:
    1/0
except ZeroDivisionError as e:
    print(e)  # division by zero
```

* * *

### **2.9 Constants**

*   `True`, `False`, `None`, `Ellipsis` (`...`), `NotImplemented`

```python
print(True, False, None, ...) 
print(NotImplemented) 
```

* * *

### **2.10 Advanced Usage of `builtins` Module**

*   **Overriding built-ins:** You can override built-in functions in your namespace, though it’s generally discouraged.

```python
import builtins

def print(x):
    builtins.print("Custom print:", x)

print("Hello")  # Custom print: Hello
builtins.print("Hello again") # Hello again
```

*   **Dynamic access:** Using `getattr` on `builtins`:

```python
import builtins
func = getattr(builtins, 'len')
print(func([1,2,3]))  # 3
```

* * *

**3\. Summary Table of Common Built-ins**
-----------------------------------------

| Category | Functions/Objects Examples |
| --- | --- |
| Type Conversion | int, float, str, bool, complex |
| Collection | len, sum, max, min, sorted, reversed |
| Iterables | enumerate, zip, map, filter, any, all |
| Object Inspection | type, isinstance, issubclass, id, dir |
| Math | abs, round, divmod, pow |
| I/O | print, input |
| Object Management | del, globals, locals, callable, getattr, setattr, hasattr |
| Constants | True, False, None, Ellipsis, NotImplemented |
| Exceptions | Exception, ValueError, TypeError, ZeroDivisionError, etc. |

* * *

**4\. Tips and Best Practices**
-------------------------------

1.  Avoid overriding Python built-ins unless necessary. Use `builtins` to access original functions if you do.
2.  Use `dir(builtins)` to discover new functions or objects.
3.  For debugging, `globals()` and `locals()` are extremely useful.
4.  Built-ins are **optimized** for performance—prefer them over custom implementations.

* * *

If you want, I can create a **“Builtins Master Cheat Sheet”** with **all built-in functions, exceptions, constants, and examples in one single document** for quick reference.

Do you want me to do that next?

## Prompt:
math module

## Response:
Perfect! Let’s do a **complete deep dive into Python’s `math` module**. This is one of the most commonly used standard modules for mathematics in Python, providing efficient functions for numbers, trigonometry, logarithms, and more.

* * *

**Python `math` Module – Complete Guide**
=========================================

**1\. Overview**
----------------

*   The `math` module provides **mathematical functions** defined by the C standard.
*   It works with **floating-point and integer numbers**, offering high performance.
*   To use it:

```python
import math
```

*   Useful for:
    *   Basic arithmetic
    *   Trigonometry
    *   Logarithms
    *   Factorials and combinatorics
    *   Constants like `pi` and `e`

* * *

**2\. Constants**
-----------------

| Constant | Description | Example |
| --- | --- | --- |
| `math.pi` | Value of π | `3.141592653589793` |
| `math.e` | Euler’s number | `2.718281828459045` |
| `math.tau` | 2π | `6.283185307179586` |
| `math.inf` | Infinity | `float('inf')` |
| `math.nan` | Not a number | `float('nan')` |

```python
import math
print(math.pi, math.e, math.tau)
print(math.inf > 1000)   # True
print(math.isnan(math.nan))  # True
```

* * *

**3\. Basic Mathematical Functions**
------------------------------------

| Function | Description | Example |
| --- | --- | --- |
| `math.sqrt(x)` | Square root | `math.sqrt(16)` → 4.0 |
| `math.pow(x, y)` | x^y | `math.pow(2, 3)` → 8.0 |
| `math.exp(x)` | e^x | `math.exp(1)` → 2.718… |
| `math.log(x[, base])` | Logarithm (default base e) | `math.log(8, 2)` → 3.0 |
| `math.log10(x)` | Base-10 log | `math.log10(100)` → 2.0 |
| `math.log2(x)` | Base-2 log | `math.log2(8)` → 3.0 |
| `math.fabs(x)` | Absolute value (float) | `math.fabs(-5)` → 5.0 |
| `math.factorial(n)` | Factorial | `math.factorial(5)` → 120 |
| `math.fsum(iterable)` | Accurate floating-point sum | `math.fsum([0.1, 0.1, 0.1])` → 0.3 |

```python
print(math.sqrt(25))       # 5.0
print(math.pow(3, 3))      # 27.0
print(math.exp(2))         # 7.38905609893065
print(math.log(16, 2))     # 4.0
print(math.factorial(6))    # 720
```

* * *

**4\. Rounding and Mod Functions**
----------------------------------

| Function | Description | Example |
| --- | --- | --- |
| `math.ceil(x)` | Round up to nearest integer | `math.ceil(3.2)` → 4 |
| `math.floor(x)` | Round down to nearest integer | `math.floor(3.8)` → 3 |
| `math.trunc(x)` | Truncate decimal part | `math.trunc(3.9)` → 3 |
| `math.modf(x)` | Returns fractional and integer parts | `math.modf(3.14)` → (0.14, 3.0) |
| `math.fmod(x, y)` | Floating-point modulo | `math.fmod(7, 3)` → 1.0 |

```python
print(math.ceil(4.2))    # 5
print(math.floor(4.9))   # 4
print(math.trunc(-4.7))  # -4
print(math.modf(5.25))   # (0.25, 5.0)
print(math.fmod(7, 2))   # 1.0
```

* * *

**5\. Trigonometric Functions**
-------------------------------

*   All angles are in **radians**. Use `math.radians(degrees)` and `math.degrees(radians)` to convert.

| Function | Description | Example |
| --- | --- | --- |
| `math.sin(x)` | Sine | `math.sin(math.pi/2)` → 1.0 |
| `math.cos(x)` | Cosine | `math.cos(0)` → 1.0 |
| `math.tan(x)` | Tangent | `math.tan(math.pi/4)` → 1.0 |
| `math.asin(x)` | Inverse sine | `math.asin(1)` → π/2 |
| `math.acos(x)` | Inverse cosine | `math.acos(1)` → 0.0 |
| `math.atan(x)` | Inverse tangent | `math.atan(1)` → π/4 |
| `math.atan2(y, x)` | Arc tangent of y/x in radians | `math.atan2(1,1)` → π/4 |

```python
angle_deg = 90
angle_rad = math.radians(angle_deg)
print(math.sin(angle_rad))  # 1.0
print(math.degrees(math.pi/2)) # 90.0
```

* * *

**6\. Hyperbolic Functions**
----------------------------

| Function | Description |
| --- | --- |
| `math.sinh(x)` | Hyperbolic sine |
| `math.cosh(x)` | Hyperbolic cosine |
| `math.tanh(x)` | Hyperbolic tangent |
| `math.asinh(x)` | Inverse hyperbolic sine |
| `math.acosh(x)` | Inverse hyperbolic cosine |
| `math.atanh(x)` | Inverse hyperbolic tangent |

```python
print(math.sinh(0))   # 0.0
print(math.cosh(0))   # 1.0
print(math.tanh(1))   # 0.7615941559557649
```

* * *

**7\. Special Functions**
-------------------------

| Function | Description | Example |
| --- | --- | --- |
| `math.erf(x)` | Error function | `math.erf(1)` → 0.8427007929497149 |
| `math.erfc(x)` | Complementary error function | `math.erfc(1)` → 0.1572992070502851 |
| `math.gamma(x)` | Gamma function | `math.gamma(5)` → 24.0 |
| `math.lgamma(x)` | Natural log of gamma | `math.lgamma(5)` → 3.1780538303479458 |

* * *

**8\. Combinatorics (Python 3.8+)**
-----------------------------------

| Function | Description | Example |
| --- | --- | --- |
| `math.comb(n, k)` | Number of combinations (n choose k) | `math.comb(5,2)` → 10 |
| `math.perm(n, k)` | Number of permutations | `math.perm(5,2)` → 20 |

```python
print(math.comb(6,3))  # 20
print(math.perm(6,3))  # 120
```

* * *

**9\. Comparison and Utility Functions**
----------------------------------------

| Function | Description | Example |
| --- | --- | --- |
| `math.isfinite(x)` | Checks if number is finite | `math.isfinite(10)` → True |
| `math.isinf(x)` | Checks if number is infinite | `math.isinf(math.inf)` → True |
| `math.isnan(x)` | Checks if not a number | `math.isnan(math.nan)` → True |
| `math.copysign(x, y)` | Copies sign of y to x | `math.copysign(3, -1)` → -3.0 |
| `math.nextafter(x, y)` | Next floating-point towards y | `math.nextafter(1.0, 2.0)` → 1.0000000000000002 |

* * *

**10\. Example Usage – Putting It All Together**
------------------------------------------------

```python
import math

# Compute hypotenuse
a = 3
b = 4
hypotenuse = math.hypot(a, b)  # sqrt(a^2 + b^2)
print(hypotenuse)  # 5.0

# Trigonometry
angle = 30
rad = math.radians(angle)
print(math.sin(rad))  # 0.5

# Factorials & combinatorics
print(math.factorial(5))   # 120
print(math.comb(5, 2))     # 10
print(math.perm(5, 2))     # 20

# Logs and exponentials
print(math.exp(2))          # 7.389...
print(math.log(8, 2))       # 3.0
```

* * *

**11\. Tips & Best Practices**
------------------------------

1.  Use `math.fsum()` for accurate floating-point sums to avoid precision issues.
2.  For angle conversions, always use `math.radians()` and `math.degrees()`.
3.  Use combinatorics functions `comb` and `perm` for efficient calculations instead of manually computing factorials.
4.  The `math` module works **only with numbers**. For complex numbers, use the `cmath` module.

* * *

If you want, I can next make a **“Math Module Master Cheat Sheet”** with **all functions, constants, and examples in one single reference table**, which is perfect for quick coding reference.

Do you want me to create that next?

## Prompt:
itertools module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `itertools` module**. This module is extremely powerful for working with **iterators** and performing combinatorial operations efficiently.

* * *

**Python `itertools` Module – Complete Guide**
==============================================

**1\. Overview**
----------------

*   `itertools` is a **standard library module** for **efficient looping and iteration**.
*   It provides **functions that return iterators**, which are **memory efficient** because they generate elements on demand instead of storing them all at once.
*   Useful for:
    *   Infinite iterators
    *   Combinatorics (permutations, combinations)
    *   Filtering and grouping
    *   Efficient looping over sequences

```python
import itertools
```

* * *

**2\. Categories of Functions in `itertools`**
----------------------------------------------

1.  **Infinite Iterators**
2.  **Iterators terminating on the shortest input sequence**
3.  **Combinatoric Iterators**

* * *

**3\. Infinite Iterators**
--------------------------

These generate values indefinitely. **Use with care**, often combined with `islice` to limit iteration.

| Function | Description | Example |
| --- | --- | --- |
| `count(start=0, step=1)` | Counts indefinitely | `itertools.count(10, 2)` → 10,12,14,... |
| `cycle(iterable)` | Loops over iterable forever | `itertools.cycle([1,2,3])` → 1,2,3,1,2,3,... |
| `repeat(object, times=None)` | Repeat object indefinitely or n times | `itertools.repeat(5, 3)` → 5,5,5 |

```python
import itertools

# count
for i in itertools.count(5, 2):
    if i > 15:
        break
    print(i)  # 5, 7, 9, 11, 13, 15

# cycle
cycled = itertools.cycle(['A','B'])
for i, val in enumerate(cycled):
    if i > 5:
        break
    print(val)  # A, B, A, B, A, B

# repeat
for val in itertools.repeat('Hello', 3):
    print(val)  # Hello Hello Hello
```

* * *

**4\. Iterators terminating on shortest input sequence**
--------------------------------------------------------

These operate **until the shortest input iterable is exhausted**.

| Function | Description | Example |
| --- | --- | --- |
| `accumulate(iterable, func=operator.add)` | Running totals (or other function) | `[1,2,3,4]` → `[1,3,6,10]` |
| `chain(*iterables)` | Combine multiple iterables | `[1,2] + [3,4]` → `[1,2,3,4]` |
| `compress(data, selectors)` | Filters data where selectors are True | `[1,2,3], [1,0,1]` → `[1,3]` |
| `dropwhile(predicate, iterable)` | Drops elements while predicate is True | `[1,2,3], lambda x:x<3` → `[3,...]` |
| `takewhile(predicate, iterable)` | Takes elements while predicate is True | `[1,2,3,4], x<3` → `[1,2]` |
| `filterfalse(predicate, iterable)` | Opposite of filter | `[1,2,3,4], x%2==0` → `[1,3]` |
| `islice(iterable, start, stop, step)` | Slice iterator | `range(10), 2,8,2` → `[2,4,6]` |
| `starmap(func, iterable)` | Map with argument tuples | `[(2,3),(3,4)], operator.pow` → `[8,81]` |
| `tee(iterable, n=2)` | Splits one iterator into n independent iterators | `it1, it2 = tee(range(3),2)` |
| `zip_longest(*iterables, fillvalue=None)` | Zip with unequal lengths, filling missing | `[1,2], [3]` → \[(1,3),(2,None)\] |

```python
import itertools, operator

# accumulate
print(list(itertools.accumulate([1,2,3,4])))  # [1,3,6,10]
print(list(itertools.accumulate([1,2,3,4], operator.mul)))  # [1,2,6,24]

# chain
print(list(itertools.chain([1,2],[3,4])))  # [1,2,3,4]

# compress
print(list(itertools.compress([1,2,3],[1,0,1])))  # [1,3]

# dropwhile & takewhile
print(list(itertools.dropwhile(lambda x:x<3,[1,2,3,4])))  # [3,4]
print(list(itertools.takewhile(lambda x:x<3,[1,2,3,4])))  # [1,2]
```

* * *

**5\. Combinatoric Iterators**
------------------------------

These generate combinations, permutations, and Cartesian products.

| Function | Description | Example |
| --- | --- | --- |
| `product(*iterables, repeat=1)` | Cartesian product | `[1,2] x ['a','b']` → \[(1,'a'),(1,'b'),(2,'a'),(2,'b')\] |
| `permutations(iterable, r=None)` | r-length permutations | `[1,2,3], 2` → \[(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)\] |
| `combinations(iterable, r)` | r-length combinations (order doesn't matter) | `[1,2,3],2` → \[(1,2),(1,3),(2,3)\] |
| `combinations_with_replacement(iterable,r)` | Combinations allowing repeated elements | `[1,2],2` → \[(1,1),(1,2),(2,2)\] |

```python
# product
print(list(itertools.product([1,2], ['a','b'])))
# [(1,'a'),(1,'b'),(2,'a'),(2,'b')]

# permutations
print(list(itertools.permutations([1,2,3], 2)))
# [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]

# combinations
print(list(itertools.combinations([1,2,3],2)))
# [(1,2),(1,3),(2,3)]

# combinations with replacement
print(list(itertools.combinations_with_replacement([1,2],2)))
# [(1,1),(1,2),(2,2)]
```

* * *

**6\. Practical Examples**
--------------------------

### **6.1 Iterating over a repeating pattern**

```python
colors = ['red', 'green', 'blue']
for color in itertools.cycle(colors):
    print(color)
    # stop manually to prevent infinite loop
```

### **6.2 Take first N elements from infinite iterator**

```python
for i in itertools.islice(itertools.count(10), 5):
    print(i)
# 10,11,12,13,14
```

### **6.3 Grouping elements**

```python
data = [1,1,2,2,3,3,3]
grouped = itertools.groupby(data)
for key, group in grouped:
    print(key, list(group))
# 1 [1,1], 2 [2,2], 3 [3,3]
```

### **6.4 Generating a deck of cards**

```python
ranks = '23456789TJQKA'
suits = 'CDHS'
deck = list(itertools.product(ranks, suits))
print(deck[:5])
# [('2','C'),('2','D'),('2','H'),('2','S'),('3','C')]
```

* * *

**7\. Tips and Best Practices**
-------------------------------

1.  **Memory efficiency:** Use `itertools` when working with large sequences because it generates items lazily.
2.  **Infinite iterators:** Always limit with `islice` or `break` to prevent infinite loops.
3.  **Combinatorics:** Avoid generating very large permutations or products directly, as it grows factorially.
4.  **Functional chaining:** Combine functions like `filterfalse`, `starmap`, and `chain` for clean, readable pipelines.
5.  **Groupby requires sorted input** for logical grouping.

* * *

If you want, I can create a **“Complete `itertools` Master Cheat Sheet”** with **all functions, examples, and common usage patterns** in one single reference document.

Do you want me to do that next?

## Prompt:
collections module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `collections` module**. This module provides **specialized container datatypes** that are extremely useful beyond the basic `list`, `dict`, `tuple`, and `set`.

* * *

**Python `collections` Module – Complete Guide**
================================================

**1\. Overview**
----------------

*   The `collections` module implements **high-performance container datatypes**.
*   It is especially useful for:
    *   Counting elements
    *   Maintaining order
    *   Handling default values
    *   Efficient queues and stacks
    *   Named tuples for readable code

```python
import collections
```

* * *

**2\. Core Classes in `collections`**
-------------------------------------

1.  **`Counter`** – Counting hashable objects
2.  **`defaultdict`** – Dictionary with default values
3.  **`OrderedDict`** – Dictionary that remembers insertion order (Python 3.7+ normal dict is ordered by default)
4.  **`namedtuple`** – Tuple with named fields
5.  **`deque`** – Double-ended queue
6.  **`ChainMap`** – Combine multiple dictionaries
7.  **`UserDict`, `UserList`, `UserString`** – Wrapper classes for customization

* * *

**3\. Counter**
---------------

*   A `Counter` counts the **frequency of elements** in an iterable.

```python
from collections import Counter

data = ['apple','banana','apple','orange','banana','apple']
c = Counter(data)
print(c)  # Counter({'apple':3,'banana':2,'orange':1})

# Most common elements
print(c.most_common(2))  # [('apple',3), ('banana',2)]

# Updating counts
c.update(['banana','banana','kiwi'])
print(c)  # Counter({'apple':3,'banana':4,'orange':1,'kiwi':1})

# Subtract counts
c.subtract(['apple','kiwi'])
print(c)  # Counter({'banana':4,'apple':2,'orange':1,'kiwi':0})
```

* * *

**4\. defaultdict**
-------------------

*   `defaultdict` provides a **default value for missing keys** instead of raising `KeyError`.

```python
from collections import defaultdict

# Default type: int
dd = defaultdict(int)
dd['a'] += 1
dd['b'] += 2
print(dd)  # defaultdict(<class 'int'>, {'a':1, 'b':2})

# Default type: list
dd_list = defaultdict(list)
dd_list['fruits'].append('apple')
dd_list['fruits'].append('banana')
print(dd_list)  # defaultdict(<class 'list'>, {'fruits':['apple','banana']})

# Default type: lambda
dd_lambda = defaultdict(lambda: 'N/A')
print(dd_lambda['unknown'])  # 'N/A'
```

* * *

**5\. OrderedDict**
-------------------

*   Maintains **insertion order** of keys (though Python 3.7+ dict already does this).
*   Provides methods like `move_to_end`.

```python
from collections import OrderedDict

od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3

print(od)  # OrderedDict([('a',1),('b',2),('c',3)])

# Move 'a' to end
od.move_to_end('a')
print(od)  # OrderedDict([('b',2),('c',3),('a',1)])
```

* * *

**6\. namedtuple**
------------------

*   A `namedtuple` is like a regular tuple but with **named fields** for readability.

```python
from collections import namedtuple

# Define a point
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20

# Convert to dict
print(p._asdict())  # {'x': 10, 'y': 20}

# Replace values
p2 = p._replace(x=100)
print(p2)  # Point(x=100, y=20)
```

* * *

**7\. deque (Double-Ended Queue)**
----------------------------------

*   `deque` allows **fast append and pop operations** from both ends.

```python
from collections import deque

dq = deque([1,2,3])
dq.append(4)      # add right
dq.appendleft(0)  # add left
print(dq)         # deque([0,1,2,3,4])

dq.pop()          # remove right
dq.popleft()      # remove left
print(dq)         # deque([1,2,3])

# Rotate
dq.rotate(1)      # right rotation
print(dq)         # deque([3,1,2])
dq.rotate(-1)     # left rotation
print(dq)         # deque([1,2,3])
```

* * *

**8\. ChainMap**
----------------

*   Combines multiple dictionaries into **one view**. Useful for configurations or variable scopes.

```python
from collections import ChainMap

dict1 = {'a':1,'b':2}
dict2 = {'b':10,'c':3}

cm = ChainMap(dict1, dict2)
print(cm['a'])  # 1
print(cm['b'])  # 2 (from first dict)
print(cm['c'])  # 3

# Add new dict
dict3 = {'d':4}
cm = cm.new_child(dict3)
print(cm['d'])  # 4
```

* * *

**9\. UserDict, UserList, UserString**
--------------------------------------

*   Wrapper classes that allow **subclassing** built-in containers easily.

```python
from collections import UserDict, UserList, UserString

class MyDict(UserDict):
    def __setitem__(self, key, value):
        print(f'Setting {key} = {value}')
        super().__setitem__(key,value)

d = MyDict()
d['a'] = 10  # Setting a = 10
```

* * *

**10\. Practical Examples**
---------------------------

### **10.1 Count occurrences**

```python
text = "hello world"
c = Counter(text)
print(c)  # Counter({'l':3,'o':2,'h':1,...})
```

### **10.2 Sliding window with deque**

```python
data = [1,2,3,4,5]
window = deque(maxlen=3)
for x in data:
    window.append(x)
    print(list(window))
# [1],[1,2],[1,2,3],[2,3,4],[3,4,5]
```

### **10.3 Grouping items by key**

```python
people = [('Alice','F'),('Bob','M'),('Clara','F')]
people.sort(key=lambda x:x[1])
from itertools import groupby
for gender, group in groupby(people, key=lambda x:x[1]):
    print(gender, list(group))
# F [('Alice','F'),('Clara','F')]
# M [('Bob','M')]
```

* * *

**11\. Tips & Best Practices**
------------------------------

1.  Use **`Counter`** for counting elements in iterables—it’s much faster than manual counting.
2.  Use **`defaultdict`** to avoid checking if keys exist.
3.  **`deque`** is better than lists for queue operations (`O(1)` for append/pop from both ends).
4.  Use **`namedtuple`** for readable and immutable records.
5.  `OrderedDict` is mostly obsolete in Python 3.7+, but still useful for `move_to_end`.

* * *

If you want, I can create a **“Collections Module Master Cheat Sheet”** with **all classes, methods, examples, and common patterns** in one reference table, which is extremely handy for coding interviews and real projects.

Do you want me to create that next?

## Prompt:
re module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `re` module**. This module is essential for **working with regular expressions**, which allows you to match, search, and manipulate text patterns efficiently.

* * *

**Python `re` Module – Complete Guide**
=======================================

**1\. Overview**
----------------

*   The `re` module provides **regular expression operations** in Python.
*   Regular expressions (regex) allow **pattern-based string searching and manipulation**.
*   Import it using:

```python
import re
```

*   **Common use cases**:
    *   Validating emails, phone numbers, or URLs
    *   Extracting information from text
    *   Replacing or splitting text based on patterns
    *   Searching for multiple variations of a string

* * *

**2\. Regex Basics**
--------------------

*   **Literal characters**: Match themselves
    ```python
    re.match('abc', 'abcdef')  # matches 'abc'
    ```
*   **Special characters**:

| Symbol | Meaning |
| --- | --- |
| `.` | Any character except newline |
| `^` | Start of string |
| `$` | End of string |
| `*` | 0 or more repetitions |
| `+` | 1 or more repetitions |
| `?` | 0 or 1 repetition |
| `{m,n}` | Between m and n repetitions |
| `[]` | Set of characters |
| \` | \` |
| `()` | Grouping |

* * *

**3\. Core Functions**
----------------------

### **3.1 `re.match(pattern, string, flags=0)`**

*   Tries to match the **pattern at the beginning** of the string.
*   Returns a **match object** or `None`.

```python
import re

m = re.match(r'\d+', '123abc')
if m:
    print(m.group())  # '123'
```

* * *

### **3.2 `re.search(pattern, string, flags=0)`**

*   Searches **anywhere in the string** for the pattern.
*   Returns a match object for the **first occurrence**.

```python
s = "My number is 4567"
m = re.search(r'\d+', s)
print(m.group())  # '4567'
```

* * *

### **3.3 `re.findall(pattern, string, flags=0)`**

*   Returns a **list of all matches** in the string.

```python
text = "cat, bat, rat"
matches = re.findall(r'\b\w{3}\b', text)
print(matches)  # ['cat', 'bat', 'rat']
```

* * *

### **3.4 `re.finditer(pattern, string, flags=0)`**

*   Returns an **iterator of match objects** for all occurrences.

```python
for m in re.finditer(r'\d+', '123 abc 456'):
    print(m.group())
# 123
# 456
```

* * *

### **3.5 `re.sub(pattern, repl, string, count=0, flags=0)`**

*   Replaces **matches with a replacement string**.

```python
text = "apples 123 oranges 456"
new_text = re.sub(r'\d+', '#', text)
print(new_text)  # 'apples # oranges #'
```

*   Using a function as replacement:

```python
def double_num(match):
    return str(int(match.group())*2)

text = "1 2 3"
print(re.sub(r'\d+', double_num, text))  # '2 4 6'
```

* * *

### **3.6 `re.split(pattern, string, maxsplit=0, flags=0)`**

*   Splits a string by **pattern matches**.

```python
text = "one,two;three four"
parts = re.split(r'[,\s;]+', text)
print(parts)  # ['one', 'two', 'three', 'four']
```

* * *

### **3.7 `re.fullmatch(pattern, string, flags=0)`**

*   Returns a match **only if the entire string matches**.

```python
m = re.fullmatch(r'\d+', '123')
print(m.group())  # '123'

m2 = re.fullmatch(r'\d+', '123abc')
print(m2)  # None
```

* * *

### **3.8 `re.compile(pattern, flags=0)`**

*   Precompiles a regex pattern for **faster repeated use**.
*   Returns a **regex object** with its own methods (`match`, `search`, `findall`, etc.)

```python
pattern = re.compile(r'\d+')
print(pattern.findall('abc 123 xyz 456'))  # ['123', '456']
```

* * *

**4\. Match Object Methods**
----------------------------

*   **`group()`** – matched text
*   **`groups()`** – tuple of all groups
*   **`start()` / `end()` / `span()`** – positions of match
*   **`group(n)`** – nth captured group

```python
m = re.search(r'(\d+)-(\w+)', '123-abc')
print(m.group())    # '123-abc'
print(m.group(1))   # '123'
print(m.group(2))   # 'abc'
print(m.groups())   # ('123','abc')
print(m.span())     # (0,7)
```

* * *

**5\. Flags**
-------------

*   Modify regex behavior using **bitwise OR (`|`)**.

| Flag | Description |
| --- | --- |
| `re.I` or `re.IGNORECASE` | Case-insensitive |
| `re.M` or `re.MULTILINE` | `^` and `$` match start/end of **lines** |
| `re.S` or `re.DOTALL` | `.` matches **newline** too |
| `re.X` or `re.VERBOSE` | Ignore whitespace and allow comments in regex |
| `re.A` or `re.ASCII` | `\w`, `\d`, `\s` match ASCII only |

```python
text = "Hello\nhello"
print(re.findall(r'^hello', text, re.I | re.M))  # ['Hello', 'hello']
```

* * *

**6\. Grouping and Capturing**
------------------------------

*   Use **parentheses `()`** to capture parts of the match.
*   Named groups: `(?P<name>pattern)`

```python
text = "2025-09-01"
m = re.match(r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})', text)
print(m.group('year'))   # '2025'
print(m.groupdict())     # {'year':'2025','month':'09','day':'01'}
```

* * *

**7\. Assertions**
------------------

| Assertion | Description |
| --- | --- |
| `^` | Start of string |
| `$` | End of string |
| `\b` | Word boundary |
| `\B` | Non-word boundary |
| `(?=...)` | Positive lookahead |
| `(?!...)` | Negative lookahead |
| `(?<=...)` | Positive lookbehind |
| `(?<!...)` | Negative lookbehind |

```python
text = "apple pie"
# Match 'apple' only if followed by space
print(re.search(r'apple(?=\s)', text))  # <re.Match object; span=(0,5)>

# Match 'pie' only if preceded by space
print(re.search(r'(?<=\s)pie', text))  # <re.Match object; span=(6,9)>
```

* * *

**8\. Practical Examples**
--------------------------

### **8.1 Validate Email**

```python
pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
emails = ["test@example.com","invalid@com"]
for email in emails:
    print(email, bool(re.fullmatch(pattern, email)))
# True, False
```

### **8.2 Extract all hashtags**

```python
text = "I love #python and #AI"
hashtags = re.findall(r'#\w+', text)
print(hashtags)  # ['#python', '#AI']
```

### **8.3 Split by multiple separators**

```python
text = "one,two;three four"
print(re.split(r'[,\s;]+', text))  # ['one','two','three','four']
```

### **8.4 Mask sensitive data**

```python
text = "My card: 1234-5678-9876-5432"
masked = re.sub(r'\d{4}-\d{4}-\d{4}-(\d{4})', r'****-****-****-\1', text)
print(masked)  # My card: ****-****-****-5432
```

* * *

**9\. Tips & Best Practices**
-----------------------------

1.  Use **raw strings `r''`** to avoid escaping backslashes.
2.  Precompile patterns with `re.compile` for repeated use.
3.  Use **`fullmatch`** for validation instead of `match`.
4.  Combine flags with `|` for flexible regex behavior.
5.  Use named groups for readability in complex patterns.
6.  Test regex patterns interactively on tools like regex101.com.

* * *

If you want, I can next create a **“Regex (`re` module) Master Cheat Sheet”** containing **all functions, symbols, assertions, and examples in one reference table**—perfect for interviews and real projects.

Do you want me to make that next?

## Prompt:
functools modules

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `functools` module**. This module provides **higher-order functions**—functions that **operate on or return other functions**—and tools for functional programming in Python.

* * *

**Python `functools` Module – Complete Guide**
==============================================

**1\. Overview**
----------------

*   The `functools` module is part of the **Python standard library**.
*   Useful for:
    *   **Decorators**
    *   **Caching results**
    *   **Partial function application**
    *   **Comparing and sorting functions**
    *   **Reducing boilerplate code**

```python
import functools
```

* * *

**2\. Core Functions and Utilities**
------------------------------------

### **2.1 `functools.partial(func, *args, **kwargs)`**

*   Creates a **new function with some arguments fixed**.
*   Useful to simplify repeated function calls with common parameters.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(2))    # 8
```

* * *

### **2.2 `functools.reduce(function, iterable[, initializer])`**

*   Applies a function **cumulatively to the items** of an iterable, reducing it to a single value.
*   Example: sum, product, factorial.

```python
from functools import reduce
import operator

numbers = [1,2,3,4]

# Sum
total = reduce(operator.add, numbers)
print(total)  # 10

# Product
product = reduce(operator.mul, numbers)
print(product)  # 24
```

* * *

### **2.3 `functools.lru_cache(maxsize=128, typed=False)`**

*   **Memoization decorator**: caches the results of a function to speed up repeated calls.
*   `maxsize=None` → cache unlimited results.

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(10))  # 55
print(fib.cache_info())  
# CacheInfo(hits=8, misses=11, maxsize=32, currsize=11)
```

* * *

### **2.4 `functools.wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)`**

*   **Decorator helper**: preserves metadata of the original function (like `__name__` and `__doc__`) when writing decorators.

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Before call")
        result = func(*args, **kwargs)
        print("After call")
        return result
    return wrapper

@my_decorator
def greet(name):
    """Greet someone"""
    print(f"Hello {name}")

greet("Alice")
print(greet.__name__)  # greet
print(greet.__doc__)   # Greet someone
```

* * *

### **2.5 `functools.cmp_to_key(func)`**

*   Converts an old-style **comparison function** (returns -1,0,1) into a **key function** usable in `sorted()`.
*   Useful when migrating Python 2 code to Python 3.

```python
from functools import cmp_to_key

def compare_length(a, b):
    return len(a) - len(b)

words = ["apple","fig","banana"]
sorted_words = sorted(words, key=cmp_to_key(compare_length))
print(sorted_words)  # ['fig', 'apple', 'banana']
```

* * *

### **2.6 `functools.total_ordering`**

*   Class decorator that fills in **missing ordering methods** if at least one comparison method (`__lt__`, `__le__`, `__gt__`, `__ge__`) is defined.

```python
from functools import total_ordering

@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    def __eq__(self, other):
        return self.grade == other.grade
    def __lt__(self, other):
        return self.grade < other.grade

s1 = Student("Alice", 90)
s2 = Student("Bob", 85)

print(s1 > s2)  # True
print(s1 <= s2) # False
```

* * *

### **2.7 `functools.singledispatch`**

*   **Function overloading** based on the **type of the first argument**.
*   Can register multiple implementations for different types.

```python
from functools import singledispatch

@singledispatch
def process(value):
    print(f"Default: {value}")

@process.register(int)
def _(value):
    print(f"Integer: {value}")

@process.register(str)
def _(value):
    print(f"String: {value}")

process(10)      # Integer: 10
process("hello") # String: hello
process([1,2])   # Default: [1,2]
```

* * *

### **2.8 `functools.cache` (Python 3.9+)**

*   Simple **unbounded cache decorator** (like `lru_cache(maxsize=None)`).

```python
from functools import cache

@cache
def factorial(n):
    return n * factorial(n-1) if n > 1 else 1

print(factorial(5))  # 120
```

* * *

### **2.9 `functools.partialmethod`**

*   Similar to `partial`, but used for **methods inside classes**.

```python
from functools import partialmethod

class Calculator:
    def multiply(self, x, y):
        return x*y
    double = partialmethod(multiply, y=2)

c = Calculator()
print(c.double(5))  # 10
```

* * *

**3\. Practical Examples**
--------------------------

### **3.1 Caching expensive computation**

```python
@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))  # 832040
```

### **3.2 Creating customized functions**

```python
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(2))    # 8
```

### **3.3 Overloading a function for multiple types**

```python
@singledispatch
def describe(obj):
    print("Unknown type")

@describe.register(list)
def _(obj):
    print("List of length", len(obj))

describe([1,2,3])   # List of length 3
describe("abc")     # Unknown type
```

* * *

**4\. Tips & Best Practices**
-----------------------------

1.  **Use `lru_cache` or `cache`** for recursive or expensive functions to improve performance.
2.  **`partial`** simplifies repeated function calls with common parameters.
3.  Use **`wraps`** whenever writing decorators to preserve function metadata.
4.  **`singledispatch`** is very handy for clean, type-based polymorphism.
5.  **`total_ordering`** reduces boilerplate for comparison methods.

* * *

If you want, I can next create a **“Functools Master Cheat Sheet”** with **all functions, decorators, examples, and practical patterns** in one reference table, which is extremely handy for coding interviews and projects.

Do you want me to make that next?

## Prompt:
operator module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `operator` module**. This module provides **function equivalents of Python’s built-in operators**, which are very useful in **functional programming**, **sorting**, **key functions**, and **higher-order functions**.

* * *

**Python `operator` Module – Complete Guide**
=============================================

**1\. Overview**
----------------

*   The `operator` module contains **function versions of Python operators**.
*   Useful when you need **functions instead of operators**, for example in `map()`, `reduce()`, `sorted()`, and `key` functions.
*   Import it using:

```python
import operator
```

*   **Example**: Instead of using `+` directly:

```python
result = operator.add(5, 3)  # 8
```

* * *

**2\. Categories of Functions**
-------------------------------

### **2.1 Arithmetic Operators**

| Function | Equivalent Operator | Example |
| --- | --- | --- |
| `operator.add(a, b)` | `a + b` | 5 + 3 → 8 |
| `operator.sub(a, b)` | `a - b` | 5 - 3 → 2 |
| `operator.mul(a, b)` | `a * b` | 5 \* 3 → 15 |
| `operator.truediv(a, b)` | `a / b` | 5 / 2 → 2.5 |
| `operator.floordiv(a, b)` | `a // b` | 5 // 2 → 2 |
| `operator.mod(a, b)` | `a % b` | 5 % 2 → 1 |
| `operator.pow(a, b)` | `a ** b` | 2 \*\* 3 → 8 |
| `operator.neg(a)` | `-a` | \-5 → -5 |
| `operator.pos(a)` | `+a` | +5 → 5 |
| `operator.abs(a)` | `abs(a)` | \-5 → 5 |

```python
import operator

print(operator.add(10,5))  # 15
print(operator.pow(2,3))   # 8
print(operator.neg(10))    # -10
```

* * *

### **2.2 Comparison Operators**

| Function | Equivalent | Example |
| --- | --- | --- |
| `operator.eq(a,b)` | `a == b` | 5 == 5 → True |
| `operator.ne(a,b)` | `a != b` | 5 != 3 → True |
| `operator.lt(a,b)` | `a < b` | 2 < 5 → True |
| `operator.le(a,b)` | `a <= b` | 5 <= 5 → True |
| `operator.gt(a,b)` | `a > b` | 7 > 3 → True |
| `operator.ge(a,b)` | `a >= b` | 5 >= 3 → True |

```python
print(operator.eq(10,10))  # True
print(operator.gt(5,8))    # False
```

* * *

### **2.3 Logical Operators**

| Function | Equivalent | Example |
| --- | --- | --- |
| `operator.and_(a,b)` | `a & b` | 5 & 3 → 1 |
| `operator.or_(a,b)` | \`a | b\` |
| `operator.xor(a,b)` | `a ^ b` | 5 ^ 3 → 6 |
| `operator.invert(a)` | `~a` | ~5 → -6 |

*   Note: These are **bitwise operations**, not `and`/`or`.

* * *

### **2.4 Sequence Operators**

| Function | Equivalent | Example |
| --- | --- | --- |
| `operator.getitem(obj, key)` | `obj[key]` | `[1,2,3][1]` → 2 |
| `operator.setitem(obj, key, value)` | `obj[key] = value` | `lst[0] = 10` |
| `operator.delitem(obj, key)` | `del obj[key]` | `del lst[1]` |
| `operator.contains(obj, item)` | `item in obj` | `3 in [1,2,3]` → True |
| `operator.indexOf(obj, item)` | `obj.index(item)` | `[1,2,3].index(2)` → 1 |

```python
lst = [10,20,30]
print(operator.getitem(lst,1))  # 20
operator.setitem(lst,1,50)
print(lst)                       # [10,50,30]
```

* * *

### **2.5 Attribute and Item Access**

*   Useful for **sorting, mapping, or key functions**.

```python
from operator import itemgetter, attrgetter

# itemgetter
data = [('Alice',25),('Bob',20),('Clara',30)]
sorted_by_age = sorted(data, key=itemgetter(1))
print(sorted_by_age)  # [('Bob',20),('Alice',25),('Clara',30)]

# attrgetter
class Person:
    def __init__(self,name,age):
        self.name = name
        self.age = age

people = [Person("Alice",25), Person("Bob",20)]
sorted_people = sorted(people, key=attrgetter('age'))
print([(p.name,p.age) for p in sorted_people])
# [('Bob',20),('Alice',25)]
```

* * *

### **2.6 Callable and Functional Utilities**

| Function | Description |
| --- | --- |
| `operator.methodcaller(name, *args, **kwargs)` | Calls a method of an object |
| `operator.itemgetter(*items)` | Returns a function to fetch items from sequences |
| `operator.attrgetter(*attrs)` | Returns a function to fetch attributes from objects |

```python
from operator import methodcaller

s = "hello world"
upper_func = methodcaller('upper')
print(upper_func(s))  # 'HELLO WORLD'
```

* * *

### **2.7 Miscellaneous**

| Function | Description |
| --- | --- |
| `operator.iadd(a,b)` | `a += b` |
| `operator.isub(a,b)` | `a -= b` |
| `operator.imul(a,b)` | `a *= b` |
| `operator.itruediv(a,b)` | `a /= b` |
| `operator.ifloordiv(a,b)` | `a //= b` |
| `operator.imod(a,b)` | `a %= b` |
| `operator.ipow(a,b)` | `a **= b` |

*   These are **in-place operators** equivalent to `+=`, `-=`, etc.

* * *

**3\. Practical Examples**
--------------------------

### **3.1 Sorting a list of tuples by multiple keys**

```python
from operator import itemgetter

data = [('Alice',25,'F'),('Bob',20,'M'),('Clara',25,'F')]
sorted_data = sorted(data, key=itemgetter(1,0))  
# Sort by age, then name
print(sorted_data)
# [('Bob',20,'M'),('Alice',25,'F'),('Clara',25,'F')]
```

### **3.2 Using attrgetter for object sorting**

```python
from operator import attrgetter

class Employee:
    def __init__(self,name,salary):
        self.name = name
        self.salary = salary

emps = [Employee('Alice',5000), Employee('Bob',7000)]
sorted_emps = sorted(emps, key=attrgetter('salary'))
print([(e.name,e.salary) for e in sorted_emps])
# [('Alice',5000),('Bob',7000)]
```

### **3.3 Using methodcaller**

```python
from operator import methodcaller

words = ['apple','banana','cherry']
upper_words = list(map(methodcaller('upper'), words))
print(upper_words)  # ['APPLE','BANANA','CHERRY']
```

### **3.4 Using functions in `reduce`**

```python
from functools import reduce
import operator

numbers = [1,2,3,4]
product = reduce(operator.mul, numbers)
print(product)  # 24
```

* * *

**4\. Tips & Best Practices**
-----------------------------

1.  Use `itemgetter` or `attrgetter` for **sorting by keys or attributes**—more efficient than `lambda`.
2.  Use `operator` functions with **`map`, `filter`, `reduce`** for cleaner code.
3.  `methodcaller` is useful for **calling object methods dynamically**.
4.  Use `operator` functions for **functional programming** or **higher-order functions**.

* * *

I can also create a **“Operator Module Master Cheat Sheet”** with **all functions, examples, and practical usage** in one compact reference—perfect for coding interviews and projects.

Do you want me to make that next?

## Prompt:
string module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `string` module**. This module provides **constants, classes, and utility functions** for working with strings, especially useful for **text processing and formatting**.

* * *

**Python `string` Module – Complete Guide**
===========================================

**1\. Overview**
----------------

*   The `string` module is part of the **Python standard library**.
*   Provides:
    *   **String constants** (like ASCII letters, digits)
    *   **Template strings**
    *   **Helper functions for text manipulation**
*   Import it using:

```python
import string
```

*   Common use cases:
    *   Generating random strings
    *   Validating characters
    *   Template-based text generation
    *   Formatting strings

* * *

**2\. String Constants**
------------------------

| Constant | Description | Example |
| --- | --- | --- |
| `string.ascii_letters` | All ASCII letters (uppercase + lowercase) | `'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'` |
| `string.ascii_lowercase` | Lowercase letters | `'abcdefghijklmnopqrstuvwxyz'` |
| `string.ascii_uppercase` | Uppercase letters | `'ABCDEFGHIJKLMNOPQRSTUVWXYZ'` |
| `string.digits` | Decimal digits | `'0123456789'` |
| `string.hexdigits` | Hexadecimal digits | `'0123456789abcdefABCDEF'` |
| `string.octdigits` | Octal digits | `'01234567'` |
| `string.punctuation` | All punctuation characters | `'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{ |
| `string.printable` | All printable characters | `'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{ |
| `string.whitespace` | All whitespace characters | `' \t\n\r\x0b\x0c'` |

```python
import string

print(string.ascii_letters)  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
print(string.digits)         # '0123456789'
print(string.punctuation)    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
```

* * *

**3\. String Template Class**
-----------------------------

*   The `Template` class allows **safe and readable string substitution**.
*   Useful for generating strings dynamically.

```python
from string import Template

t = Template("Hello, $name! You have $count messages.")
s = t.substitute(name="Alice", count=5)
print(s)  # "Hello, Alice! You have 5 messages."
```

*   **Safe substitution** (doesn’t throw error if placeholder is missing):

```python
s = t.safe_substitute(name="Bob")
print(s)  # "Hello, Bob! You have $count messages."
```

* * *

**4\. Common Utility Functions**
--------------------------------

The `string` module also provides **functions that mostly act on templates or formatting**, although most text methods are in the `str` class itself.

1.  **`string.capwords(s, sep=None)`**
    *   Capitalizes the first letter of each word.
    *   Equivalent to: `' '.join(word.capitalize() for word in s.split())`

```python
import string
text = "hello world from python"
print(string.capwords(text))  # "Hello World From Python"
```

* * *

**5\. Practical Examples**
--------------------------

### **5.1 Generate a random string**

```python
import string, random

letters = string.ascii_letters
digits = string.digits
all_chars = letters + digits

rand_string = ''.join(random.choice(all_chars) for _ in range(10))
print(rand_string)  # e.g., 'aZ3bT9kLm1'
```

### **5.2 Validate characters**

```python
password = "Abc123!"
if all(c in string.ascii_letters + string.digits + string.punctuation for c in password):
    print("Valid characters")
```

### **5.3 Using Template for emails**

```python
from string import Template

template = Template("Dear $name, your balance is $balance USD.")
msg = template.substitute(name="Alice", balance=100)
print(msg)  # "Dear Alice, your balance is 100 USD."
```

### **5.4 Splitting and capitalizing words**

```python
sentence = "python string module"
capitalized = string.capwords(sentence)
print(capitalized)  # "Python String Module"
```

### **5.5 Formatting with Template**

```python
data = {'item':'laptop','price':1200}
template = Template("The price of $item is $$${price}.")
print(template.substitute(data))  # "The price of laptop is $1200."
```

* * *

**6\. Tips & Best Practices**
-----------------------------

1.  Use **string constants** instead of hardcoding letters, digits, or punctuation.
2.  Use **`Template`** for **user-facing strings** to avoid code injection risks.
3.  Use **`string.capwords`** for capitalizing titles or headings.
4.  Combine with `random` module for **passwords or random strings**.
5.  For more complex formatting, prefer **f-strings or `str.format()`**, but `Template` is safer for user-provided data.

* * *

If you want, I can next create a **“String Module Master Cheat Sheet”** containing **all constants, classes, functions, examples, and practical patterns** in one compact reference—perfect for coding interviews and projects.

Do you want me to make that next?

## Prompt:
calendar module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `calendar` module**. This module provides classes and functions to **work with dates, calendars, and weekdays**, which is useful for scheduling, date calculations, and displaying calendars.

* * *

**Python `calendar` Module – Complete Guide**
=============================================

**1\. Overview**
----------------

*   The `calendar` module is part of the **Python standard library**.
*   It provides:
    *   Calendar-related functions
    *   Classes for **iterating over months and years**
    *   Utilities to determine **leap years and weekdays**
*   Import it using:

```python
import calendar
```

*   Common use cases:
    *   Displaying monthly or yearly calendars
    *   Calculating weekdays
    *   Determining leap years
    *   Customizing first weekday

* * *

**2\. Calendar Constants**
--------------------------

| Constant | Description |
| --- | --- |
| `calendar.MONDAY` | 0 |
| `calendar.TUESDAY` | 1 |
| `calendar.WEDNESDAY` | 2 |
| `calendar.THURSDAY` | 3 |
| `calendar.FRIDAY` | 4 |
| `calendar.SATURDAY` | 5 |
| `calendar.SUNDAY` | 6 |
| `calendar.firstweekday()` | Returns current first weekday (default: Monday) |
| `calendar.setfirstweekday(weekday)` | Sets the first day of the week |

```python
print(calendar.firstweekday())  # 0 (Monday)
calendar.setfirstweekday(calendar.SUNDAY)
print(calendar.firstweekday())  # 6 (Sunday)
```

* * *

**3\. Basic Calendar Functions**
--------------------------------

### **3.1 Check Leap Year**

```python
print(calendar.isleap(2024))  # True
print(calendar.isleap(2025))  # False
```

### **3.2 Count Leap Years in a Range**

```python
print(calendar.leapdays(2000, 2025))  # Number of leap years from 2000 to 2024
```

### **3.3 Weekday of a Date**

*   `calendar.weekday(year, month, day)` → returns 0 (Monday) to 6 (Sunday)

```python
print(calendar.weekday(2025, 9, 1))  # 0 (Monday)
```

*   `calendar.day_name` → Names of the days

```python
import calendar
print(calendar.day_name[calendar.weekday(2025, 9, 1)])  # 'Monday'
```

*   `calendar.day_abbr` → Abbreviated names

```python
print(calendar.day_abbr[calendar.weekday(2025, 9, 1)])  # 'Mon'
```

* * *

**4\. Displaying Calendars**
----------------------------

### **4.1 Month Calendar**

*   Returns a **matrix (list of weeks)** where each week is a list of days (0 = no day).

```python
print(calendar.monthcalendar(2025, 9))
# Example Output:
# [[0, 1, 2, 3, 4, 5, 6],
#  [7, 8, 9,10,11,12,13],
#  [14,15,16,17,18,19,20],
#  [21,22,23,24,25,26,27],
#  [28,29,30,0,0,0,0]]
```

### **4.2 Print Month (Text Calendar)**

```python
print(calendar.month(2025,9))
```

Output:

```
   September 2025
Mo Tu We Th Fr Sa Su
 1  2  3  4  5  6  7
 8  9 10 11 12 13 14
15 16 17 18 19 20 21
22 23 24 25 26 27 28
29 30
```

### **4.3 Print Year Calendar**

```python
print(calendar.calendar(2025))
```

*   Returns **yearly calendar** as a string in text format.

* * *

**5\. HTML Calendar**
---------------------

*   Useful for **web applications**.

```python
html_cal = calendar.HTMLCalendar(calendar.SUNDAY)
print(html_cal.formatmonth(2025, 9))
```

*   This outputs a **month calendar as HTML table**.

* * *

**6\. Iterating over Days and Months**
--------------------------------------

### **6.1 Iterate Days in a Month**

```python
for day in calendar.itermonthdays(2025, 9):
    print(day, end=' ')
# 0 1 2 3 ... 30 0 0 0
```

*   `0` represents days outside the current month.

### **6.2 Iterate Weekdays**

```python
for week in calendar.monthdays2calendar(2025, 9):
    print(week)
# Each day is a tuple (day, weekday_number)
```

### **6.3 Iterate Month Names**

```python
print(list(calendar.month_name))  # ['', 'January', 'February', ... 'December']
print(list(calendar.month_abbr))  # ['', 'Jan', 'Feb', ... 'Dec']
```

* * *

**7\. Custom Calendars**
------------------------

*   **TextCalendar** and **HTMLCalendar** can be subclassed for customization.

```python
tc = calendar.TextCalendar(calendar.SUNDAY)
print(tc.formatmonth(2025, 9))
```

*   Methods:
    *   `formatmonth(year, month)` → text
    *   `formatyear(year)` → text
    *   `prmonth(year, month)` → prints month
    *   `pryear(year)` → prints year

* * *

**8\. Practical Examples**
--------------------------

### **8.1 Get all Mondays in a month**

```python
c = calendar.Calendar(firstweekday=calendar.MONDAY)
mondays = [day for day, weekday in c.itermonthdays2(2025, 9) if weekday == calendar.MONDAY and day != 0]
print(mondays)  # [1, 8, 15, 22, 29]
```

### **8.2 Check leap year and days**

```python
year = 2024
if calendar.isleap(year):
    print(f"{year} is a leap year with 366 days")
else:
    print(f"{year} is not a leap year")
```

### **8.3 Custom Text Calendar**

```python
tc = calendar.TextCalendar(calendar.SUNDAY)
print(tc.formatmonth(2025, 12))
```

### **8.4 Generate HTML Calendar**

```python
hc = calendar.HTMLCalendar(calendar.MONDAY)
html_calendar = hc.formatyear(2025)
print(html_calendar[:500])  # Print first 500 chars
```

* * *

**9\. Tips & Best Practices**
-----------------------------

1.  Use **`monthcalendar()`** or **`itermonthdays()`** for programmatically iterating days.
2.  Use **`TextCalendar`** for console output and **`HTMLCalendar`** for web.
3.  Always use `calendar.MONDAY` or `calendar.SUNDAY` for **firstweekday**—it’s more readable than 0 or 6.
4.  Combine with `datetime` for advanced scheduling or date arithmetic.
5.  Use **`leapdays()`** to count leap years in a range.

* * *

I can next create a **“Calendar Module Master Cheat Sheet”** containing **all functions, classes, constants, and examples** in one handy reference—perfect for interviews or projects.

Do you want me to make that next?

## Prompt:
datetime module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `datetime` module**. This module is essential for **working with dates, times, and timestamps**, including **arithmetic, formatting, parsing, and timezone handling**.

* * *

**Python `datetime` Module – Complete Guide**
=============================================

**1\. Overview**
----------------

*   Part of Python’s **standard library**.
*   Provides classes for **date and time manipulation**:
    *   `date` – calendar date (year, month, day)
    *   `time` – time (hour, minute, second, microsecond)
    *   `datetime` – combination of date and time
    *   `timedelta` – difference between dates or times
    *   `tzinfo` – timezone info
*   Import it using:

```python
import datetime
from datetime import date, time, datetime, timedelta
```

*   Common use cases:
    *   Tracking timestamps
    *   Calculating durations
    *   Formatting dates for display
    *   Timezone-aware datetime handling

* * *

**2\. `date` Class**
--------------------

Represents a **calendar date** (year, month, day).

### **2.1 Creating a Date**

```python
d = date(2025, 9, 1)
print(d)  # 2025-09-01
```

### **2.2 Current Date**

```python
today = date.today()
print(today)  # e.g., 2025-09-01
```

### **2.3 Attributes**

```python
print(today.year)   # 2025
print(today.month)  # 9
print(today.day)    # 1
print(today.weekday())   # 0 (Monday=0)
print(today.isoweekday())# 1 (Monday=1)
```

### **2.4 Methods**

*   `date.fromisoformat('YYYY-MM-DD')`
*   `date.strftime(format)` – format date
*   `date.replace(year=, month=, day=)` – create new date

```python
d2 = date.fromisoformat('2025-12-25')
print(d2.strftime("%A, %B %d, %Y"))  # Thursday, December 25, 2025
```

* * *

**3\. `time` Class**
--------------------

Represents a **time of day** (hour, minute, second, microsecond).

```python
t = time(14,30,15)
print(t)  # 14:30:15
print(t.hour, t.minute, t.second)  # 14 30 15
```

*   Formatting with `strftime`:

```python
print(t.strftime("%I:%M %p"))  # 02:30 PM
```

* * *

**4\. `datetime` Class**
------------------------

Represents **date and time** together.

### **4.1 Creating a datetime**

```python
dt = datetime(2025,9,1,14,30,15)
print(dt)  # 2025-09-01 14:30:15
```

### **4.2 Current DateTime**

```python
now = datetime.now()
print(now)
```

*   `datetime.today()` – current local datetime
*   `datetime.utcnow()` – current UTC datetime

### **4.3 Attributes**

```python
print(now.year, now.month, now.day)
print(now.hour, now.minute, now.second)
```

### **4.4 Formatting (`strftime`)**

| Directive | Meaning |
| --- | --- |
| `%Y` | Year with century |
| `%y` | Year without century |
| `%m` | Month (01-12) |
| `%d` | Day of month (01-31) |
| `%H` | Hour (24h) |
| `%I` | Hour (12h) |
| `%M` | Minute |
| `%S` | Second |
| `%p` | AM/PM |
| `%A` | Weekday full name |
| `%a` | Weekday abbreviated |
| `%B` | Month full name |
| `%b` | Month abbreviated |

```python
print(now.strftime("%A, %B %d, %Y %I:%M %p"))
# Monday, September 01, 2025 02:30 PM
```

### **4.5 Parsing (`strptime`)**

```python
dt_str = "2025-09-01 14:30:15"
dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
print(dt)  # 2025-09-01 14:30:15
```

* * *

**5\. `timedelta` Class**
-------------------------

Represents **duration** between dates/times.

### **5.1 Creating a timedelta**

```python
delta = timedelta(days=5, hours=3, minutes=30)
print(delta)  # 5 days, 3:30:00
```

### **5.2 Date arithmetic**

```python
today = date.today()
future = today + timedelta(days=10)
past = today - timedelta(weeks=2)
print(future, past)
```

```python
dt = datetime.now()
dt_future = dt + timedelta(hours=5, minutes=20)
print(dt_future)
```

### **5.3 Difference between dates**

```python
d1 = date(2025, 9, 1)
d2 = date(2025, 12, 25)
diff = d2 - d1
print(diff.days)  # 115
```

* * *

**6\. Timezone Support**
------------------------

Python 3.2+ supports **timezone-aware datetimes** using `tzinfo` or `datetime.timezone`.

```python
from datetime import timezone, timedelta

# UTC offset +5:30
ist = timezone(timedelta(hours=5, minutes=30))
dt = datetime.now(ist)
print(dt)  # 2025-09-01 14:30:15+05:30
```

*   Convert between timezones using `astimezone()`.

```python
utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
ist_dt = utc_dt.astimezone(ist)
print(ist_dt)
```

* * *

**7\. Practical Examples**
--------------------------

### **7.1 Calculate age from birthdate**

```python
birth = date(1995, 6, 18)
today = date.today()
age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
print(age)
```

### **7.2 Countdown to a future date**

```python
event = datetime(2025, 12, 25, 10, 0, 0)
now = datetime.now()
remaining = event - now
print(f"Days: {remaining.days}, Seconds: {remaining.seconds}")
```

### **7.3 Format current datetime for log file**

```python
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"log_{log_time}.txt"
print(filename)  # e.g., log_20250901_143015.txt
```

### **7.4 Generate a list of dates**

```python
start = date(2025,9,1)
dates = [start + timedelta(days=i) for i in range(7)]
print(dates)
# [2025-09-01, 2025-09-02, ... 2025-09-07]
```

* * *

**8\. Tips & Best Practices**
-----------------------------

1.  Use `datetime` for combined date and time, `date` for only dates.
2.  Prefer **timezone-aware datetime** for real-world applications.
3.  Use `strftime` for formatting output, `strptime` for parsing.
4.  Use `timedelta` for arithmetic instead of manually calculating days.
5.  Always be cautious with **naive vs timezone-aware datetimes**.

* * *

I can next create a **“Datetime Module Master Cheat Sheet”** with **all classes, methods, constants, formatting codes, and examples** in one compact reference—perfect for coding interviews and projects.

Do you want me to make that next?

## Prompt:
xml module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `xml` module**. Python provides several `xml` modules in its standard library to **parse, create, and manipulate XML data**.

* * *

**Python `xml` Module – Complete Guide**
========================================

**1\. Overview**
----------------

*   The `xml` package in Python contains modules to handle **XML parsing, generation, and validation**.
*   Common modules:
    1.  `xml.etree.ElementTree` – Lightweight XML parser and builder.
    2.  `xml.dom` – Document Object Model (DOM) API.
    3.  `xml.sax` – SAX parser for event-driven XML parsing.
    4.  `xml.parsers.expat` – Low-level XML parser.
*   Import the submodules as needed:

```python
import xml.etree.ElementTree as ET
from xml.dom import minidom
import xml.sax
```

*   Use cases:
    *   Reading/writing XML files
    *   Processing configuration files
    *   Converting XML data to Python objects
    *   Validating XML documents

* * *

**2\. `xml.etree.ElementTree` (ET) – Most Common**
--------------------------------------------------

### **2.1 Parsing XML**

```python
import xml.etree.ElementTree as ET

xml_data = """<root>
  <person id="1">
    <name>Alice</name>
    <age>25</age>
  </person>
  <person id="2">
    <name>Bob</name>
    <age>30</age>
  </person>
</root>"""

root = ET.fromstring(xml_data)  # Parse from string
print(root.tag)  # 'root'

for person in root.findall('person'):
    name = person.find('name').text
    age = person.find('age').text
    print(name, age)
# Alice 25
# Bob 30
```

*   **From file**:

```python
tree = ET.parse('file.xml')
root = tree.getroot()
```

* * *

### **2.2 Creating XML**

```python
root = ET.Element("root")
person = ET.SubElement(root, "person", id="1")
ET.SubElement(person, "name").text = "Alice"
ET.SubElement(person, "age").text = "25"

tree = ET.ElementTree(root)
tree.write("output.xml")
```

*   This generates:

```xml
<root>
  <person id="1">
    <name>Alice</name>
    <age>25</age>
  </person>
</root>
```

* * *

### **2.3 Modifying XML**

```python
for person in root.findall('person'):
    age = int(person.find('age').text)
    person.find('age').text = str(age + 1)  # Increment age
```

### **2.4 Pretty Print XML**

```python
import xml.dom.minidom

xml_str = ET.tostring(root)
pretty_xml = xml.dom.minidom.parseString(xml_str).toprettyxml()
print(pretty_xml)
```

* * *

**3\. `xml.dom.minidom` – DOM API**
-----------------------------------

*   Provides **tree-based API** similar to JavaScript DOM.
*   Useful for **navigating and editing XML documents**.

```python
from xml.dom import minidom

xml_data = "<root><item>One</item><item>Two</item></root>"
dom = minidom.parseString(xml_data)
items = dom.getElementsByTagName("item")
for item in items:
    print(item.firstChild.nodeValue)
# One
# Two
```

*   **Create XML with DOM**:

```python
doc = minidom.Document()
root = doc.createElement("root")
doc.appendChild(root)

item = doc.createElement("item")
item_text = doc.createTextNode("One")
item.appendChild(item_text)
root.appendChild(item)

print(doc.toprettyxml())
```

* * *

**4\. `xml.sax` – Event-Driven Parser**
---------------------------------------

*   **SAX (Simple API for XML)** parses XML **sequentially**, firing events for elements.
*   Useful for **large XML files**.

```python
import xml.sax

class MyHandler(xml.sax.ContentHandler):
    def startElement(self, name, attrs):
        print("Start:", name)
    def endElement(self, name):
        print("End:", name)
    def characters(self, content):
        if content.strip():
            print("Text:", content)

xml_data = "<root><item>One</item><item>Two</item></root>"
xml.sax.parseString(xml_data, MyHandler())
```

* * *

**5\. `xml.parsers.expat` – Low-Level Parser**
----------------------------------------------

*   Low-level parser, rarely used directly.
*   Usually, prefer `ElementTree` or `SAX`.

```python
import xml.parsers.expat

def start_element(name, attrs):
    print('Start:', name, attrs)

def end_element(name):
    print('End:', name)

parser = xml.parsers.expat.ParserCreate()
parser.StartElementHandler = start_element
parser.EndElementHandler = end_element

parser.Parse("<root><item id='1'/></root>")
```

* * *

**6\. XPath-like Queries in ElementTree**
-----------------------------------------

*   `find()`, `findall()`, `findtext()`
*   Support simple paths:

```python
root.findall('person/name')  # All names under person
```

*   Attributes:

```python
for p in root.findall('person'):
    print(p.attrib['id'])
```

* * *

**7\. Practical Examples**
--------------------------

### **7.1 Reading XML**

```python
tree = ET.parse('people.xml')
root = tree.getroot()

for person in root.findall('person'):
    print(person.find('name').text, person.find('age').text)
```

### **7.2 Writing XML**

```python
root = ET.Element("company")
ET.SubElement(root, "employee", id="1").text = "Alice"
ET.ElementTree(root).write("company.xml")
```

### **7.3 Updating XML**

```python
for emp in root.findall('employee'):
    emp.text = emp.text.upper()
```

### **7.4 Converting XML to Dictionary**

```python
def xml_to_dict(elem):
    return {elem.tag: {child.tag: child.text for child in elem}}

d = xml_to_dict(root)
print(d)
```

* * *

**8\. Tips & Best Practices**
-----------------------------

1.  Use **`ElementTree`** for most tasks—simple and efficient.
2.  Use **`SAX`** for **large XML files** (memory-efficient).
3.  Use **`minidom`** for **pretty-printing or DOM manipulation**.
4.  Always handle **missing elements** using `find()` carefully.
5.  For production-level XML validation, consider **`lxml`** (third-party, faster, XPath/XSLT support).

* * *

I can next create a **“XML Module Master Cheat Sheet”** with **all submodules, classes, methods, and practical examples** in one reference—ideal for coding interviews and projects.

Do you want me to make that next?

## Prompt:
os module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `os` module**. This module is **one of the most important built-in modules** for interacting with the **operating system**, file system, and environment.

* * *

**Python `os` Module – Complete Guide**
=======================================

**1\. Overview**
----------------

*   The `os` module provides a **portable way of using operating system-dependent functionality**.
*   Use cases:
    *   File and directory manipulation
    *   Environment variables
    *   Process management
    *   Path operations (in combination with `os.path`)
*   Import it using:

```python
import os
```

* * *

**2\. Environment Variables**
-----------------------------

| Function | Description |
| --- | --- |
| `os.environ` | Dictionary of environment variables |
| `os.getenv('VAR')` | Get the value of an environment variable |
| `os.putenv('VAR', 'value')` | Set an environment variable (changes persist only in Python process) |
| `os.environ['VAR'] = 'value'` | Recommended way to set env variable |

```python
print(os.environ['HOME'])
print(os.getenv('PATH'))

os.environ['MY_VAR'] = '123'
print(os.environ['MY_VAR'])
```

* * *

**3\. Current Working Directory (CWD)**
---------------------------------------

| Function | Description |
| --- | --- |
| `os.getcwd()` | Get current working directory |
| `os.chdir(path)` | Change current working directory |

```python
print(os.getcwd())  # e.g., /Users/username
os.chdir('/tmp')
print(os.getcwd())  # /tmp
```

* * *

**4\. File and Directory Operations**
-------------------------------------

| Function | Description |
| --- | --- |
| `os.listdir(path='.')` | List all files and directories in path |
| `os.mkdir(path)` | Create a new directory |
| `os.makedirs(path)` | Create directories recursively |
| `os.rmdir(path)` | Remove empty directory |
| `os.removedirs(path)` | Remove directories recursively if empty |
| `os.rename(src, dst)` | Rename file or directory |
| `os.remove(path)` | Remove file |
| `os.path.exists(path)` | Check if path exists |
| `os.path.isdir(path)` | Check if path is a directory |
| `os.path.isfile(path)` | Check if path is a file |

```python
os.mkdir('test_dir')
print(os.listdir('.'))
os.rename('test_dir', 'new_dir')
os.rmdir('new_dir')
```

* * *

**5\. Path Operations (`os.path`)**
-----------------------------------

| Function | Description |
| --- | --- |
| `os.path.join(a,b,...)` | Join paths correctly |
| `os.path.split(path)` | Split into `(dir, file)` |
| `os.path.splitext(path)` | Split into `(file, extension)` |
| `os.path.basename(path)` | Get file name |
| `os.path.dirname(path)` | Get directory name |
| `os.path.abspath(path)` | Absolute path |
| `os.path.relpath(path, start)` | Relative path |
| `os.path.normpath(path)` | Normalize path (remove redundant separators) |

```python
path = os.path.join('/tmp', 'file.txt')
print(path)  # '/tmp/file.txt'

print(os.path.basename(path))  # 'file.txt'
print(os.path.dirname(path))   # '/tmp'
print(os.path.splitext(path))  # ('/tmp/file', '.txt')
```

* * *

**6\. Process Management**
--------------------------

| Function | Description |
| --- | --- |
| `os.system(command)` | Execute shell command |
| `os.getpid()` | Get current process ID |
| `os.getppid()` | Get parent process ID |
| `os.startfile(path)` | Open file (Windows only) |
| `os.exec*()` | Replace current process with new process |

```python
print(os.getpid())
os.system('echo Hello World')
```

* * *

**7\. Walking Directories**
---------------------------

*   `os.walk()` – generates file names by walking the directory tree.

```python
for dirpath, dirnames, filenames in os.walk('.'):
    print('Directory:', dirpath)
    print('Subdirs:', dirnames)
    print('Files:', filenames)
```

* * *

**8\. File Permissions & Info**
-------------------------------

| Function | Description |
| --- | --- |
| `os.chmod(path, mode)` | Change permissions |
| `os.stat(path)` | Get file info (size, modified time, etc.) |
| `os.access(path, mode)` | Check access: `os.R_OK`, `os.W_OK`, `os.X_OK` |

```python
os.chmod('file.txt', 0o644)  # Owner read/write, group/world read
info = os.stat('file.txt')
print(info.st_size)  # File size
```

* * *

**9\. Temporary Directories & Files**
-------------------------------------

*   Use `os` with `tempfile` for secure temp files.

```python
import tempfile

temp_dir = tempfile.mkdtemp()
print(temp_dir)

temp_file = tempfile.NamedTemporaryFile(delete=False)
print(temp_file.name)
```

* * *

**10\. Practical Examples**
---------------------------

### **10.1 List all `.txt` files in current directory**

```python
txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
print(txt_files)
```

### **10.2 Recursive directory listing**

```python
for dirpath, dirnames, filenames in os.walk('.'):
    for file in filenames:
        print(os.path.join(dirpath, file))
```

### **10.3 Create nested directories safely**

```python
os.makedirs('dir1/dir2/dir3', exist_ok=True)
```

### **10.4 Get file info**

```python
file_info = os.stat('file.txt')
print(f"Size: {file_info.st_size} bytes")
print(f"Modified: {file_info.st_mtime}")
```

### **10.5 Execute shell command**

```python
os.system('ls -l')
```

* * *

**11\. Tips & Best Practices**
------------------------------

1.  Prefer **`os.path`** for path handling instead of string concatenation.
2.  Use **`os.makedirs(..., exist_ok=True)`** to avoid errors for existing directories.
3.  Combine **`os.walk`** with file filtering for recursive operations.
4.  Avoid `os.system()` for critical commands; prefer `subprocess` module.
5.  Always check file existence before operations using **`os.path.exists()`**.

* * *

I can next create a **“OS Module Master Cheat Sheet”** with **all functions, methods, path operations, and practical examples** in one reference—perfect for coding interviews and scripting projects.

Do you want me to make that next?

## Prompt:
sys module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `sys` module**. The `sys` module provides **access to variables and functions used or maintained by the Python interpreter**, which is essential for **system-level programming, command-line arguments, and runtime info**.

* * *

**Python `sys` Module – Complete Guide**
========================================

**1\. Overview**
----------------

*   The `sys` module is part of Python’s **standard library**.
*   Provides:
    *   Access to interpreter **variables and functions**
    *   Command-line arguments
    *   Standard input/output/error streams
    *   Python path manipulation
    *   Exiting the program
*   Import it using:

```python
import sys
```

*   Common use cases:
    *   Command-line scripts
    *   Accessing Python environment info
    *   Exiting with status codes
    *   Modifying module search path

* * *

**2\. Command-Line Arguments**
------------------------------

*   `sys.argv` – list of command-line arguments.

```python
import sys

print(sys.argv)  # ['script.py', 'arg1', 'arg2']

# Accessing arguments
if len(sys.argv) > 1:
    print("First argument:", sys.argv[1])
```

* * *

**3\. Standard Input, Output, Error**
-------------------------------------

| Attribute | Description |
| --- | --- |
| `sys.stdin` | Standard input stream |
| `sys.stdout` | Standard output stream |
| `sys.stderr` | Standard error stream |

```python
sys.stdout.write("Hello, World\n")
sys.stderr.write("Error: Something went wrong\n")
```

*   Useful for **redirecting outputs**:

```python
with open('output.txt', 'w') as f:
    sys.stdout = f
    print("This goes to the file")
```

* * *

**4\. Exiting the Program**
---------------------------

*   `sys.exit([arg])` – exit interpreter

```python
import sys

if len(sys.argv) < 2:
    print("Usage: script.py <name>")
    sys.exit(1)  # Exit with error code
```

* * *

**5\. Interpreter Information**
-------------------------------

| Function/Attribute | Description |
| --- | --- |
| `sys.version` | Python version string |
| `sys.version_info` | Tuple of major, minor, micro, releaselevel, serial |
| `sys.platform` | Platform identifier (`win32`, `linux`, `darwin`) |
| `sys.executable` | Path to Python interpreter |
| `sys.prefix` | Python installation prefix |

```python
print(sys.version)
print(sys.version_info)
print(sys.platform)
print(sys.executable)
```

* * *

**6\. Module Search Path**
--------------------------

*   `sys.path` – list of directories Python searches for modules

```python
import sys
print(sys.path)

# Add custom module path
sys.path.append('/my/custom/path')
```

* * *

**7\. Reference Counting & Garbage Collection**
-----------------------------------------------

| Function | Description |
| --- | --- |
| `sys.getrefcount(object)` | Get reference count of object |
| `sys.getsizeof(object)` | Get size in bytes |
| `sys.getallocatedblocks()` | Number of memory blocks allocated by Python |

```python
a = [1,2,3]
print(sys.getrefcount(a))  # Includes temporary reference
print(sys.getsizeof(a))    # Memory size in bytes
```

* * *

**8\. Standard I/O Encoding**
-----------------------------

*   Useful for encoding/decoding issues:

```python
print(sys.getdefaultencoding())  # Usually 'utf-8'
print(sys.getfilesystemencoding())
```

* * *

**9\. Inspecting Modules and Recursion Limit**
----------------------------------------------

| Function/Attribute | Description |
| --- | --- |
| `sys.modules` | Dictionary of loaded modules |
| `sys.getrecursionlimit()` | Current recursion limit |
| `sys.setrecursionlimit(limit)` | Set recursion limit |

```python
import sys
print(sys.modules.keys())

sys.setrecursionlimit(2000)
print(sys.getrecursionlimit())
```

* * *

**10\. Practical Examples**
---------------------------

### **10.1 Command-line arguments**

```python
# script.py
import sys
for i, arg in enumerate(sys.argv):
    print(f"Argument {i}: {arg}")
```

### **10.2 Redirect stdout to file**

```python
import sys

with open('output.txt', 'w') as f:
    sys.stdout = f
    print("This goes to file")
```

### **10.3 Exit with error code**

```python
import sys

if len(sys.argv) < 2:
    sys.exit("Error: Missing arguments")
```

### **10.4 Custom module path**

```python
import sys
sys.path.append('/my/custom/modules')
import mymodule
```

### **10.5 Check Python version**

```python
import sys
if sys.version_info < (3, 10):
    sys.exit("Python 3.10+ is required")
```

* * *

**11\. Tips & Best Practices**
------------------------------

1.  Use `sys.argv` for command-line tools.
2.  Prefer `sys.exit(code)` instead of `exit()`.
3.  Avoid permanently changing `sys.path`; use virtual environments.
4.  Use `sys.stderr` for error messages in scripts.
5.  Use `sys.version_info` for version-specific compatibility.

* * *

I can next create a **“Sys Module Master Cheat Sheet”** with **all functions, attributes, examples, and best practices** in one compact reference—perfect for Python scripting and interviews.

Do you want me to make that next?

## Prompt:
html.parser module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `html.parser` module**. This module is part of the **Python standard library** and provides tools for **parsing and handling HTML content**.

* * *

**Python `html.parser` Module – Complete Guide**
================================================

**1\. Overview**
----------------

*   `html.parser` is a **simple and lightweight HTML parser**.
*   Provides the `HTMLParser` class which can **parse HTML documents** and trigger **callbacks for tags, data, and comments**.
*   Use cases:
    *   Extracting text or links from HTML
    *   Processing web page content
    *   Creating custom HTML processing tools
*   Import it using:

```python
from html.parser import HTMLParser
```

* * *

**2\. Basic Usage**
-------------------

*   You need to **subclass `HTMLParser`** and override callback methods to handle different HTML elements.

```python
from html.parser import HTMLParser

class MyParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start tag:", tag)
        print("Attributes:", attrs)
        
    def handle_endtag(self, tag):
        print("End tag:", tag)
        
    def handle_data(self, data):
        print("Data:", data)
        
    def handle_comment(self, data):
        print("Comment:", data)
        
    def handle_entityref(self, name):
        print("Named entity:", name)
        
    def handle_charref(self, name):
        print("Numeric entity:", name)

html = """
<html>
  <body>
    <h1>Hello World</h1>
    <!-- This is a comment -->
    <p>Paragraph &copy; 2025</p>
  </body>
</html>
"""

parser = MyParser()
parser.feed(html)
```

*   Output example:

```
Start tag: html
Start tag: body
Start tag: h1
Data: Hello World
End tag: h1
Comment:  This is a comment 
Start tag: p
Data: Paragraph © 2025
End tag: p
End tag: body
End tag: html
Named entity: copy
```

* * *

**3\. Key Methods / Callbacks**
-------------------------------

| Method | Description |
| --- | --- |
| `handle_starttag(tag, attrs)` | Called for each start tag; `attrs` is a list of tuples `(name, value)` |
| `handle_endtag(tag)` | Called for each end tag |
| `handle_startendtag(tag, attrs)` | Called for self-closing tags `<br />` |
| `handle_data(data)` | Called for text data between tags |
| `handle_comment(data)` | Called for HTML comments `<!-- comment -->` |
| `handle_entityref(name)` | Called for named entities `&copy;` |
| `handle_charref(name)` | Called for numeric entities `&#169;` |
| `handle_decl(decl)` | Called for `<!DOCTYPE ...>` declarations |
| `handle_pi(data)` | Called for processing instructions `<? ... ?>` |

* * *

**4\. Working with Attributes**
-------------------------------

*   `attrs` is a **list of `(key, value)` tuples**:

```python
class LinkParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, val in attrs:
                if attr == 'href':
                    print("Link found:", val)

html = '<a href="https://example.com">Example</a>'
parser = LinkParser()
parser.feed(html)
```

Output:

```
Link found: https://example.com
```

* * *

**5\. Handling HTML Entities**
------------------------------

*   Named entity: `&copy;`, `&amp;`
*   Numeric entity: `&#169;` or `&#xA9;`

```python
class EntityParser(HTMLParser):
    def handle_entityref(self, name):
        print("Named entity:", name)
    def handle_charref(self, name):
        print("Numeric entity:", name)

parser = EntityParser()
parser.feed("Copyright &copy; 2025 &#169;")
```

Output:

```
Named entity: copy
Numeric entity: 169
```

* * *

**6\. Parsing Links and Data Example**
--------------------------------------

```python
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr, val in attrs:
                if attr == "href":
                    self.links.append(val)

    def handle_data(self, data):
        self.text.append(data.strip())

html = """
<p>Welcome to <a href="https://example.com">Example</a> website.</p>
<p>Contact us <a href="mailto:info@example.com">here</a>.</p>
"""

parser = MyHTMLParser()
parser.feed(html)
print("Links:", parser.links)
print("Text:", [t for t in parser.text if t])
```

Output:

```
Links: ['https://example.com', 'mailto:info@example.com']
Text: ['Welcome to', 'Example', 'website.', 'Contact us', 'here', '.']
```

* * *

**7\. Advanced Usage**
----------------------

### **7.1 Handle Self-Closing Tags**

```python
class MyParser(HTMLParser):
    def handle_startendtag(self, tag, attrs):
        print("Self-closing tag:", tag)

html = '<img src="image.png" />'
parser = MyParser()
parser.feed(html)
```

### **7.2 Resetting Parser**

```python
parser.reset()  # Reset parser state
parser.close()  # Close parser and process remaining data
```

### **7.3 Converting to Text**

```python
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return ''.join(self.text)

html = "<p>Hello <b>World</b></p>"
extractor = TextExtractor()
extractor.feed(html)
print(extractor.get_text())  # "Hello World"
```

* * *

**8\. Tips & Best Practices**
-----------------------------

1.  Use `html.parser` for **lightweight parsing of well-formed HTML**.
2.  For **complex or broken HTML**, prefer `BeautifulSoup` with `"html.parser"` backend.
3.  Always **subclass `HTMLParser`** and override required methods.
4.  Use `handle_data` to extract text and `handle_starttag` for attributes.
5.  Reset parser with `.reset()` if you plan to reuse it.

* * *

I can next create a **“HTMLParser Module Master Cheat Sheet”** with **all methods, callbacks, examples, and practical patterns** in one compact reference—perfect for web scraping and text extraction tasks.

Do you want me to make that next?

## Prompt:
exceptions module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python’s `exceptions`**.

> Note: Python does not have a separate `exceptions` module; exceptions are built-in classes in Python. The **`builtins` module** contains all exception classes. So here, we’ll cover **Python exception hierarchy, handling, and best practices** in depth.

* * *

**Python Exceptions – Complete Guide**
======================================

**1\. Overview**
----------------

*   Exceptions are **runtime errors** or unusual situations during execution.
*   Python has **built-in exception classes** organized in a hierarchy.
*   Use cases:
    *   Handle runtime errors gracefully
    *   Control program flow in error scenarios
    *   Logging or debugging errors

* * *

**2\. Exception Hierarchy (Key Classes)**
-----------------------------------------

```
BaseException
 ├── SystemExit
 ├── KeyboardInterrupt
 ├── GeneratorExit
 └── Exception
      ├── ArithmeticError
      │    ├── FloatingPointError
      │    ├── OverflowError
      │    └── ZeroDivisionError
      ├── AssertionError
      ├── AttributeError
      ├── EOFError
      ├── ImportError
      │     └── ModuleNotFoundError
      ├── LookupError
      │     ├── IndexError
      │     └── KeyError
      ├── MemoryError
      ├── NameError
      │     └── UnboundLocalError
      ├── OSError
      │     ├── FileNotFoundError
      │     ├── PermissionError
      │     └── TimeoutError
      ├── RuntimeError
      │     ├── NotImplementedError
      │     └── RecursionError
      ├── StopIteration
      ├── TypeError
      └── ValueError
```

* * *

**3\. Basic Exception Handling**
--------------------------------

### **3.1 `try-except`**

```python
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
```

### **3.2 Multiple Exceptions**

```python
try:
    val = int("abc")
except (ValueError, TypeError) as e:
    print("Error:", e)
```

### **3.3 Generic Exception**

```python
try:
    x = 1 / 0
except Exception as e:
    print("Exception occurred:", e)
```

* * *

**4\. `else` and `finally` Blocks**
-----------------------------------

*   **`else`** runs if no exception occurs
*   **`finally`** always runs

```python
try:
    x = 10 / 2
except ZeroDivisionError:
    print("Division by zero")
else:
    print("Division succeeded:", x)
finally:
    print("This always runs")
```

* * *

**5\. Raising Exceptions**
--------------------------

*   Use `raise` to throw exceptions.

```python
def divide(a, b):
    if b == 0:
        raise ValueError("b cannot be zero")
    return a / b

try:
    divide(5, 0)
except ValueError as e:
    print("Caught:", e)
```

* * *

**6\. Custom Exceptions**
-------------------------

*   Create **user-defined exceptions** by subclassing `Exception`.

```python
class MyError(Exception):
    pass

def test(value):
    if value < 0:
        raise MyError("Value must be non-negative")

try:
    test(-1)
except MyError as e:
    print("Custom Exception:", e)
```

* * *

**7\. Common Built-in Exceptions**
----------------------------------

| Exception | When it occurs |
| --- | --- |
| `ZeroDivisionError` | Division by zero |
| `TypeError` | Operation on invalid type |
| `ValueError` | Invalid value for function |
| `IndexError` | Index out of range |
| `KeyError` | Dictionary key not found |
| `AttributeError` | Invalid attribute access |
| `FileNotFoundError` | File not found |
| `ImportError` | Module import failed |
| `ModuleNotFoundError` | Module does not exist |
| `NameError` | Undefined variable |
| `RuntimeError` | Unspecified runtime error |
| `RecursionError` | Maximum recursion exceeded |

* * *

**8\. Exception Chaining**
--------------------------

*   Preserve original exception when raising a new one:

```python
try:
    int("abc")
except ValueError as e:
    raise RuntimeError("Conversion failed") from e
```

*   `from None` suppresses chaining:

```python
try:
    int("abc")
except ValueError:
    raise RuntimeError("Conversion failed") from None
```

* * *

**9\. Practical Examples**
--------------------------

### **9.1 File Handling with Exceptions**

```python
try:
    with open("data.txt", "r") as f:
        data = f.read()
except FileNotFoundError:
    print("File does not exist")
except IOError:
    print("Error reading file")
```

### **9.2 Safe Division Function**

```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

print(safe_divide(10, 2))  # 5.0
print(safe_divide(10, 0))  # None
```

### **9.3 Handling Multiple Exceptions**

```python
try:
    x = int("abc")
    y = 1 / 0
except ValueError:
    print("Invalid conversion")
except ZeroDivisionError:
    print("Cannot divide by zero")
```

* * *

**10\. Best Practices**
-----------------------

1.  Catch **specific exceptions** rather than generic `Exception`.
2.  Use `finally` to **release resources**.
3.  Prefer **custom exceptions** for application-specific errors.
4.  Avoid using exceptions for **normal control flow**.
5.  Chain exceptions with `raise ... from ...` for **better debugging**.

* * *

I can next create a **“Python Exceptions Master Cheat Sheet”** with **all built-in exceptions, hierarchy, usage, and examples** in one compact reference—perfect for debugging and interviews.

Do you want me to make that next?

## Prompt:
decorators module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python decorators**.

> Note: Python does **not have a built-in `decorators` module**. Decorators are a **language feature** implemented with functions, classes, and the `functools` module. We’ll cover everything: function decorators, class decorators, built-in decorators, and advanced usage.

* * *

**Python Decorators – Complete Guide**
======================================

**1\. Overview**
----------------

*   **Decorators** are functions (or classes) that **modify or extend the behavior of another function, method, or class**.
*   Use cases:
    *   Logging
    *   Caching
    *   Validation
    *   Timing functions
    *   Access control (authentication/authorization)
*   Syntax:

```python
@decorator
def func():
    pass
```

*   Equivalent to:

```python
def func():
    pass
func = decorator(func)
```

* * *

**2\. Function Decorators**
---------------------------

### **2.1 Basic Example**

```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

Output:

```
Before function call
Hello!
After function call
```

* * *

### **2.2 Decorators with Arguments**

```python
def decorator_with_args(func):
    def wrapper(*args, **kwargs):
        print("Arguments:", args, kwargs)
        return func(*args, **kwargs)
    return wrapper

@decorator_with_args
def add(a, b):
    return a + b

print(add(5, 3))  # Arguments: (5,3) {}   => 8
```

* * *

### **2.3 Decorators Returning Values**

```python
def multiply_result(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@multiply_result
def add(a, b):
    return a + b

print(add(5, 3))  # 16
```

* * *

**3\. Built-in Decorators**
---------------------------

### **3.1 `@staticmethod`**

*   Defines a method that **does not receive `self`**.

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

print(Math.add(2, 3))  # 5
```

### **3.2 `@classmethod`**

*   Defines a method that **receives class `cls` instead of instance**.

```python
class Person:
    count = 0
    
    def __init__(self, name):
        self.name = name
        Person.count += 1

    @classmethod
    def total(cls):
        return cls.count

p1 = Person("Alice")
p2 = Person("Bob")
print(Person.total())  # 2
```

### **3.3 `@property`**

*   Allows **getter/setter style access** like attributes.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        self._radius = r

c = Circle(5)
print(c.radius)
c.radius = 10
print(c.radius)
```

* * *

**4\. Decorators in `functools`**
---------------------------------

### **4.1 `functools.wraps`**

*   Preserves **original function’s metadata** (`__name__`, `__doc__`) when wrapping.

```python
from functools import wraps

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Before call")
        return func(*args, **kwargs)
    return wrapper

@decorator
def greet():
    """Say hello"""
    print("Hello")

print(greet.__name__)  # greet
print(greet.__doc__)   # Say hello
```

* * *

### **4.2 `lru_cache`**

*   Built-in **caching decorator**.

```python
from functools import lru_cache

@lru_cache(maxsize=3)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(10))  # 55
```

* * *

**5\. Class Decorators**
------------------------

*   Decorators can also **wrap classes**.

```python
def add_str_method(cls):
    cls.__str__ = lambda self: f"MyClass with value={self.value}"
    return cls

@add_str_method
class MyClass:
    def __init__(self, value):
        self.value = value

obj = MyClass(10)
print(obj)  # MyClass with value=10
```

* * *

**6\. Decorators with Parameters**
----------------------------------

*   Decorator factory: returns a decorator.

```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello")

say_hello()
```

Output:

```
Hello
Hello
Hello
```

* * *

**7\. Nested Decorators**
-------------------------

```python
def bold(func):
    def wrapper(*args, **kwargs):
        return "<b>" + func(*args, **kwargs) + "</b>"
    return wrapper

def italic(func):
    def wrapper(*args, **kwargs):
        return "<i>" + func(*args, **kwargs) + "</i>"
    return wrapper

@bold
@italic
def greet():
    return "Hello"

print(greet())  # <b><i>Hello</i></b>
```

*   Execution order: **bottom-up** (closest decorator executes first).

* * *

**8\. Practical Examples**
--------------------------

### **8.1 Logging Decorator**

```python
def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args}, {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    return a + b

add(2, 3)
```

### **8.2 Timing Decorator**

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end-start:.5f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

slow_function()
```

* * *

**9\. Tips & Best Practices**
-----------------------------

1.  Use `@wraps` to preserve metadata.
2.  Keep decorators **simple and composable**.
3.  Use **decorator factories** for parameters.
4.  Avoid **overusing nested decorators** for readability.
5.  Use `functools.lru_cache` for caching expensive computations.
6.  Prefer function decorators for **behavioral changes**, class decorators for **structural changes**.

* * *

I can next create a **“Python Decorators Master Cheat Sheet”** with **all types of decorators, examples, patterns, and built-in decorators** in one compact reference—perfect for coding interviews and production projects.

Do you want me to make that next?

## Prompt:
classes module

## Response:
Perfect! Let’s do a **complete, in-depth guide to Python classes**, often referred to as the “classes module” in practical terms. Python doesn’t have a separate `classes` module—**classes are a core language feature**—but I’ll cover **everything from basic to advanced class concepts, including metaclasses and built-in class tools**.

* * *

**Python Classes – Complete Guide**
===================================

**1\. Overview**
----------------

*   Classes are **blueprints for creating objects** (instances).
*   Each object has **attributes (data) and methods (functions)**.
*   Use cases:
    *   Encapsulation of data and behavior
    *   Code reuse through inheritance
    *   Modeling real-world objects

```python
class MyClass:
    pass

obj = MyClass()
```

* * *

**2\. Defining a Class**
------------------------

```python
class Person:
    species = "Homo sapiens"  # Class attribute

    def __init__(self, name, age):  # Constructor
        self.name = name          # Instance attribute
        self.age = age

    def greet(self):  # Instance method
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Create instance
p1 = Person("Alice", 25)
p1.greet()  # Hello, my name is Alice and I am 25 years old.
```

* * *

**3\. Class vs Instance Attributes**
------------------------------------

```python
p2 = Person("Bob", 30)
print(p1.species)  # Homo sapiens
print(p2.species)  # Homo sapiens

Person.species = "Human"
print(p1.species)  # Human
print(p2.species)  # Human
```

*   **Instance attributes** are unique to each object (`self.name`)
*   **Class attributes** are shared across instances (`Person.species`)

* * *

**4\. Methods in Python Classes**
---------------------------------

| Method Type | Definition | Example |
| --- | --- | --- |
| Instance Method | Default, first arg `self` | `def greet(self)` |
| Class Method | Decorator `@classmethod`, first arg `cls` | `@classmethod def info(cls)` |
| Static Method | Decorator `@staticmethod`, no `self` | `@staticmethod def utility()` |
| Special Methods | Magic methods (`__init__`, `__str__`, `__repr__`) | `def __str__(self)` |

* * *

### **4.1 Class Method**

```python
class Person:
    count = 0

    def __init__(self, name):
        self.name = name
        Person.count += 1

    @classmethod
    def total(cls):
        return cls.count

p1 = Person("Alice")
p2 = Person("Bob")
print(Person.total())  # 2
```

### **4.2 Static Method**

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

print(Math.add(5, 3))  # 8
```

### **4.3 Magic Methods**

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def __str__(self):
        return f"Circle with radius {self.radius}"

    def __add__(self, other):
        return Circle(self.radius + other.radius)

c1 = Circle(5)
c2 = Circle(10)
c3 = c1 + c2
print(c3)  # Circle with radius 15
```

* * *

**5\. Inheritance**
-------------------

*   **Single inheritance**

```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Dog barks")

d = Dog()
d.speak()  # Dog barks
```

*   **Multiple inheritance**

```python
class Flyer:
    def fly(self):
        print("Can fly")

class Bird(Animal, Flyer):
    pass

b = Bird()
b.speak()  # Animal speaks
b.fly()    # Can fly
```

*   **`super()`** – call parent methods

```python
class Dog(Animal):
    def speak(self):
        super().speak()
        print("Dog barks")

Dog().speak()
```

* * *

**6\. Encapsulation (Private / Protected)**
-------------------------------------------

*   **Private attribute**: double underscore `__var`
*   **Protected attribute**: single underscore `_var`

```python
class Person:
    def __init__(self, name):
        self._name = name    # Protected
        self.__age = 25      # Private

p = Person("Alice")
print(p._name)      # Accessible
# print(p.__age)    # Error
print(p._Person__age)  # Access private using name mangling
```

* * *

**7\. Properties (`@property`)**
--------------------------------

*   Create **getter/setter for attributes**

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        if r < 0:
            raise ValueError("Radius must be positive")
        self._radius = r

c = Circle(5)
c.radius = 10
print(c.radius)
```

* * *

**8\. Advanced: Metaclasses**
-----------------------------

*   A **metaclass is a class of a class**, controlling class creation.
*   Default metaclass: `type`

```python
class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass
# Output: Creating class MyClass
```

*   Use metaclasses for **custom class creation, validation, or decorators at class level**.

* * *

**9\. Class Introspection**
---------------------------

```python
class Person:
    species = "Human"

print(type(Person))       # <class 'type'>
print(dir(Person))        # List attributes/methods
print(hasattr(Person, 'species'))  # True
print(getattr(Person, 'species'))  # Human
```

* * *

**10\. Practical Examples**
---------------------------

### **10.1 Employee Class with Inheritance**

```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def show_info(self):
        print(f"{self.name}: ${self.salary}")

class Manager(Employee):
    def show_info(self):
        super().show_info()
        print("Role: Manager")

m = Manager("Alice", 5000)
m.show_info()
```

### **10.2 Singleton Pattern**

```python
class Singleton:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

### **10.3 Using Properties**

```python
class Temperature:
    def __init__(self, celsius):
        self._c = celsius

    @property
    def fahrenheit(self):
        return self._c * 9/5 + 32

t = Temperature(25)
print(t.fahrenheit)  # 77.0
```

* * *

**11\. Tips & Best Practices**
------------------------------

1.  Use **`__init__` for instance initialization**.
2.  Prefer **composition over multiple inheritance**.
3.  Use **`@property`** for controlled attribute access.
4.  Keep class **methods and responsibilities small and clear**.
5.  Use **metaclasses sparingly**; mainly for frameworks.
6.  Use **`super()`** to ensure proper parent method calls.
7.  Encapsulate private data using **underscore conventions**.

* * *

I can next create a **“Python Classes Master Cheat Sheet”** with **all concepts: class types, inheritance, methods, magic methods, metaclasses, examples, and patterns** in one compact reference—perfect for interviews and production coding.

Do you want me to make that next?

## Prompt:
Give a deep cheat sheet of this entire chat

2/2

## Response:
Got it ✅  
Here’s a **deep Python Modules + Classes Cheat Sheet** covering **everything we discussed in this chat (builtins → decorators → classes)**. It’s structured for **quick reference + interview prep**.

* * *

🐍 **Python Modules & Classes Cheat Sheet**
===========================================

* * *

🔹 **1\. `builtins` Module**
----------------------------

*   Provides **core functions, exceptions, types**.
*   Examples:

```python
abs(-5)        # 5
len([1,2,3])   # 3
max([1,5,3])   # 5
min([1,5,3])   # 1
sum([1,2,3])   # 6
round(3.1415, 2)  # 3.14
```

*   Types: `int`, `float`, `str`, `list`, `tuple`, `dict`, `set`
*   Exceptions: `ValueError`, `TypeError`, `IndexError`, `KeyError`

* * *

🔹 **2\. `math` Module**
------------------------

*   Math functions:

```python
import math
math.sqrt(16)   # 4.0
math.factorial(5)  # 120
math.gcd(24, 36)   # 12
math.ceil(2.3)     # 3
math.floor(2.9)    # 2
math.pi            # 3.1415926535
```

* * *

🔹 **3\. `itertools` Module**
-----------------------------

*   Iterator building tools:

```python
from itertools import count, cycle, repeat, chain, permutations, combinations, groupby

list(count(5, 2))[:5]   # [5, 7, 9, 11, 13]
list(cycle("AB")[:6])   # A B A B A B
list(repeat(10, 3))     # [10, 10, 10]
list(chain([1,2], [3,4]))  # [1,2,3,4]
list(permutations([1,2,3], 2))  # [(1,2),(1,3),(2,1)...]
list(combinations([1,2,3], 2))  # [(1,2),(1,3),(2,3)]
```

* * *

🔹 **4\. `collections` Module**
-------------------------------

*   Specialized data structures:

```python
from collections import Counter, defaultdict, deque, namedtuple

Counter("banana")   # Counter({'a':3,'n':2,'b':1})
d = defaultdict(int); d["x"]+=1  # auto default
dq = deque([1,2]); dq.appendleft(0); dq.pop()
Point = namedtuple("Point", "x y"); p = Point(1,2)
```

* * *

🔹 **5\. `re` Module**
----------------------

*   Regex operations:

```python
import re
re.match(r"ab", "abc")     # match at start
re.search(r"b", "abc")     # finds first
re.findall(r"[a-z]", "abc") # ['a','b','c']
re.sub(r"[0-9]", "*", "a1b2")  # a*b*
```

* * *

🔹 **6\. `functools` Module**
-----------------------------

```python
from functools import reduce, lru_cache, wraps

reduce(lambda x,y: x+y, [1,2,3])  # 6

@lru_cache(maxsize=3)
def fib(n): return n if n<2 else fib(n-1)+fib(n-2)

def decorator(func):
    @wraps(func)
    def wrapper(*a,**kw): return func(*a,**kw)
    return wrapper
```

* * *

🔹 **7\. `operator` Module**
----------------------------

```python
import operator as op
op.add(2,3)     # 5
op.sub(5,2)     # 3
op.mul(2,3)     # 6
op.truediv(5,2) # 2.5
op.eq(2,2)      # True
op.itemgetter(1)([10,20,30])  # 20
op.attrgetter("real")(3+4j)   # 3.0
```

* * *

🔹 **8\. `string` Module**
--------------------------

```python
import string
string.ascii_letters   # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
string.digits          # '0123456789'
string.punctuation     # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
template = string.Template("Hello $name")
template.substitute(name="Alice")  # "Hello Alice"
```

* * *

🔹 **9\. `calendar` Module**
----------------------------

```python
import calendar
print(calendar.month(2025, 9))     # September 2025
calendar.isleap(2024)              # True
calendar.weekday(2025, 9, 11)      # 3 (Thursday)
```

* * *

🔹 **10\. `datetime` Module**
-----------------------------

```python
from datetime import datetime, timedelta, date
now = datetime.now()
today = date.today()
delta = timedelta(days=5)
print(now + delta)
```

* * *

🔹 **11\. `xml` Module**
------------------------

```python
import xml.etree.ElementTree as ET
root = ET.Element("root")
child = ET.SubElement(root, "child")
child.text = "data"
ET.dump(root)
```

* * *

🔹 **12\. `os` Module**
-----------------------

```python
import os
os.name             # 'posix' or 'nt'
os.getcwd()         # current dir
os.listdir(".")     # list files
os.environ["PATH"]  # env variable
os.system("echo hi")
```

* * *

🔹 **13\. `sys` Module**
------------------------

```python
import sys
print(sys.version)
print(sys.platform)
sys.argv            # command-line args
sys.exit(0)
```

* * *

🔹 **14\. `html.parser` Module**
--------------------------------

```python
from html.parser import HTMLParser
class MyParser(HTMLParser):
    def handle_starttag(self, tag, attrs): print("Start:", tag)
    def handle_data(self, data): print("Data:", data)

parser = MyParser()
parser.feed("<h1>Hello</h1>")
```

* * *

🔹 **15\. Exceptions**
----------------------

*   Built-in exceptions: `ValueError`, `TypeError`, `IndexError`, `ZeroDivisionError`, `KeyError`

```python
try:
    x = 1/0
except ZeroDivisionError:
    print("Division by zero!")
finally:
    print("Done")
```

* * *

🔹 **16\. Decorators**
----------------------

*   **Basic**

```python
def my_decorator(func):
    def wrapper(): print("Before"); func(); print("After")
    return wrapper

@my_decorator
def hello(): print("Hello")
```

*   **With args**

```python
def repeat(n):
    def decorator(func):
        def wrapper(*a,**k): [func(*a,**k) for _ in range(n)]
        return wrapper
    return decorator

@repeat(3)
def hi(): print("Hi")
```

*   Built-in: `@staticmethod`, `@classmethod`, `@property`
*   From `functools`: `@wraps`, `@lru_cache`

* * *

🔹 **17\. Classes**
-------------------

### **Basic Class**

```python
class Person:
    species = "Human"
    def __init__(self, name): self.name = name
    def greet(self): print(f"Hi, I'm {self.name}")

p = Person("Alice"); p.greet()
```

### **Inheritance**

```python
class Dog:
    def speak(self): print("Bark")

class Puppy(Dog):
    def speak(self): super().speak(); print("Whine")
```

### **Encapsulation**

```python
class Demo:
    def __init__(self): self._protected=1; self.__private=2
```

### **Properties**

```python
class Circle:
    def __init__(self,r): self._r=r
    @property def radius(self): return self._r
    @radius.setter def radius(self,v): self._r=v
```

### **Special Methods**

```python
class Point:
    def __init__(self,x,y): self.x=x; self.y=y
    def __str__(self): return f"({self.x},{self.y})"
    def __add__(self,o): return Point(self.x+o.x,self.y+o.y)
```

### **Singleton**

```python
class Singleton:
    _instance=None
    def __new__(cls,*a,**k):
        if not cls._instance: cls._instance=super().__new__(cls)
        return cls._instance
```

### **Metaclass**

```python
class Meta(type):
    def __new__(cls,name,bases,dct):
        print("Creating",name); return super().__new__(cls,name,bases,dct)

class MyClass(metaclass=Meta): pass
```

* * *

✅ **Tips & Best Practices**
===========================

*   Always use `functools.wraps` in decorators.
*   Prefer `@property` instead of getter/setter methods.
*   Use `lru_cache` for expensive recursive functions.
*   Keep classes small, single-responsibility.
*   Avoid multiple inheritance unless needed (prefer composition).
*   Use `sys.argv` for command-line programs.
*   Use `collections` for performance over built-in lists/dicts when needed.

* * *

⚡This cheat sheet is **your one-stop guide** for Python **modules + decorators + classes** from **basic → advanced**.

Would you like me to also create this in a **visual structured PDF cheat sheet (tables + diagrams + code snippets)** for easier study and printing?



---
