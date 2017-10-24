---
layout: post
title:  "Introduction to Python Programming"
date:   2017-09-21 1:31:56 +0530
categories: lesson
---

# Numbers, Strings, and Lists

Python supports a number of built-in types and operations. For more information  [click here](https://docs.python.org/3/library/stdtypes.html).

## Basic numeric types

**Integers (``int``)**


```python
i = 1
j = 219089
k = -21231
```


```python
print(i, j, k)
```

Note that this is a point where python2 and python3 differ:  python2 will
output
    (1, 219089, -21231)
here, python3
    1 219089 -21231
In the jargon: python3 has made print a function, where it's been a statement in python2.

-> Here the brackets had a different meaning in python2.

**Floating point values (``float``)**


```python
a = 4.3
b = -5.2111222
c = 3.1e33
```


```python
print(a, b, c)
```

    4.3 -5.2111222 3.1e+33


Manipulating these behaves the way you would expect, so an operation (``+``, ``-``, ``*``, ``**``, etc.) on two values of the same type produces another value of the same type (with one, exception, ``/``, see below), while an operation on two values with different types produces a value of the more 'advanced' type:

Adding two integers gives an integer:


```python
1 + 3
```




    4



Multiplying two floats gives a float:


```python
3. * 2.
```




    6.0



Multiplying an integer with a float gives a float:


```python
3 * 9.2
```




    27.599999999999998



However, the division of two integers gives a float:


```python
3 / 2
```




    1.5



Note that in Python 2.x, this used to return ``1`` because it would round the solution to an integer. If you ever need to work with Python 2 code, the safest approach is to add the following line at the top of the script:

    from __future__ import division
    
and the division will then behave like a Python 3 division. Note that in Python 3 you can also specifically request integer division:


```python
3 // 2
```




    1



## Strings

Strings (``str``) are sequences of characters:


```python
s = "Spam egg spam spam"
```

You can use either single quotes (``'``), double quotes (``"``), or triple quotes (``'''`` or ``"""``) to enclose a string (the last one is used for multi-line strings). To include single or double quotes inside a string, you can either use the opposite quote to enclose the string:



```python
"I'm"
```




    "I'm"




```python
'"hello"'
```




    '"hello"'



or you can *escape* them:


```python
'I\'m'
```




    "I'm"




```python
"\"hello\""
```




    '"hello"'



You can access individual characters or chunks of characters using the item notation with square brackets``[]``:


```python
s[5]
```




    'e'



Note that in Python, indexing is *zero-based*, which means that the first element in a list is zero:


```python
s[0]
```




    'S'



Note that strings are **immutable**, that is you cannot change the value of certain characters without creating a new string:


```python
s[5] = 'r'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-21-e268c53f8105> in <module>()
    ----> 1 s[5] = 'r'
    

    TypeError: 'str' object does not support item assignment


You can easily find the length of a string:


```python
len(s)
```




    18



You can use the ``+`` operator to combine strings:


```python
"hello," + " " + "world!"
```




    'hello, world!'



Finally, strings have many **methods** associated with them, here are a few examples:


```python
s.upper()  # An uppercase version of the string
```




    'SPAM EGG SPAM SPAM'




```python
s.index('egg')  # An integer giving the position of the sub-string
```




    5




```python
s.split()  # A list of strings
```




    ['Spam', 'egg', 'spam', 'spam']



## Lists

There are several kinds of ways of storing sequences in Python, the simplest being the ``list``, which is simply a sequence of *any* Python object.


```python
li = [4, 5.5, "spam"]
```

Accessing individual items is done like for strings


```python
li[0]
```




    4




```python
li[1]
```




    5.5




```python
li[2]
```




    'spam'



Values in a list can be changed, and it is also possible to append or insert elements:


```python
li[1] = -2.2
```


```python
li
```




    [4, -2.2, 'spam']




```python
li.append(-3)
```


```python
li
```




    [4, -2.2, 'spam', -3]




```python
li.insert(1, 3.14) # insert at position 1
```


```python
li
```




    [4, 3.14, -2.2, 'spam', -3]



As for strings, you can find the length of a list (the number of elements) with the ``len`` function:


```python
len([1,2,3,4,5])
```




    5



## Slicing

We already mentioned above that it is possible to access individual elements from a string or a list using the square bracket notation. You will also find this notation for other object types in Python, for example tuples or Numpy arrays, so it's worth spending a bit of time looking at this in more detail.

In addition to using positive integers, where ``0`` is the first item, it is possible to access list items with *negative* indices, which counts from the end: ``-1`` is the last element, ``-2`` is the second to last, etc:


```python
li = [4, 67, 4, 2, 4, 6]
```


```python
li[-1]
```




    6



You can also select **slices** from a list with the ``start:end:step`` syntax. Be aware that the last element is *not* included!


```python
li[0:2]
```




    [4, 67]




```python
li[:2]  # ``start`` defaults to zero
```




    [4, 67]




```python
li[2:]  # ``end`` defaults to the last element 
```




    [4, 2, 4, 6]




```python
li[::2]  # specify a step size
```




    [4, 4, 4]



## Exercise 2

Given a string such as the one below, make a new string that does not contain the word ``egg``:


```python
a = "Hello, egg world!"

# enter your solution here

```

Try changing the string above to see if your solution works (you can assume that ``egg`` appears only once in the string).

## A note on Python objects (demo)

Most things in Python are objects.  But what is an object?

Every constant, variable, or function in Python is actually a object with a
type and associated attributes and methods. An *attribute* a property of the
object that you get or set by giving the ``<object_name>.<attribute_name>``, for example ``img.shape``. A *method* is a function that the object provides, for example ``img.argmax(axis=0)`` or ``img.min()``.
    
Use tab completion in IPython to inspect objects and start to understand
attributes and methods. To start off create a list of 4 numbers:

    li = [3, 1, 2, 1]
    li.<TAB>

This will show the available attributes and methods for the Python list
``li``.

**Using ``<TAB>``-completion and help is a very efficient way to learn and later
remember object methods!**

    In [2]: li.
    li.append   li.copy     li.extend   li.insert   li.remove   li.sort
    li.clear    li.count    li.index    li.pop      li.reverse 
    
If you want to know what a function or method does, you can use a question mark ``?``:
    
    In [9]: li.append?
    Type:       builtin_function_or_method
    String Form:<built-in method append of list object at 0x1027210e0>
    Docstring:  L.append(object) -> None -- append object to end

## Dynamic typing

One final note on Python types: if you know languages like C, Java or, say, Pascal, you will remember you had to declare variables.  Essentially, types are bound to names in such “statically typed” languages.  In Python, on the other hand, types sit on the values, and names just point there; there's nothing that keeps you from having a single name point to differently typed values (except perhaps common sense in may cases).  Jargon calls this *dynamic typing*.


```python
a = 1
type(a)
```




    int




```python
a = 2.3
type(a)
```




    float




```python
a = 'hello'
type(a)
```




    str



## Converting between types

There may be cases where you want to convert a string to a floating point value, and integer to a string, etc. For this, you can simply use the ``int()``, ``float()``, and ``str()`` functions:


```python
int('1')
```




    1




```python
float('4.31')
```




    4.31



For example:


```python
int('5') + float('4.31')
```




    9.309999999999999



is different from:


```python
'5' + '4.31'
```

Similarly:


```python
str(1)
```


```python
str(4.5521)
```


```python
str(3) + str(4)
```

Be aware of this for example when connecting strings with numbers, as you can only concatenate identical types this way:


```python
'The value is ' + 3
```

Instead do:


```python
'The value is ' + str(3)
```


# Booleans, Tuples, and Dictionaries

## Booleans

A ``boolean`` is one of the simplest Python types, and it can have two values: ``True`` and ``False`` (with uppercase ``T`` and ``F``):


```python
a = True
b = False
```

Booleans can be combined with logical operators to give other booleans:


```python
True and False
```


```python
True or False
```


```python
(False and (True or False)) or (False and True)
```

Standard comparison operators can also produce booleans:


```python
1 == 3
```


```python
1 != 3
```


```python
3 > 2
```


```python
3 <= 3.4
```

## Exercise 1

Write an expression that returns ``True`` if ``x`` is strictly greater than 3.4 and smaller or equal to 6.6, or if it is 2, and try changing ``x`` to see if it works:


```python
x = 3.7

# your solution here

```

## Tuples

Tuples are, like lists, a type of sequence, but they use round parentheses rather than square brackets:


```python
t = (1, 2, 3)
```

They can contain heterogeneous types like lists:


```python
t = (1, 2.3, 'spam')
```

and also support item access and slicing like lists:


```python
t[1]
```


```python
t[:2]
```

The main difference is that they are **immutable**, like strings:


```python
t[1] = 2
```

## Dictionaries

One of the data types that we have not talked about yet is called *dictionaries* (``dict``). If you think about what a 'real' dictionary is, it is a list of words, and for each word is a definition. Similarly, in Python, we can assign definitions (or 'values'), to words (or 'keywords').

Dictionaries are defined using curly brackets ``{}``:


```python
d = {'a':1, 'b':2, 'c':3}
```

Items are accessed using square brackets and the 'key':


```python
d['a']
```


```python
d['c']
```

Values can also be set this way:


```python
d['r'] = 2.2
```


```python
print(d)
```

The keywords don't have to be strings, they can be many (but not all) Python objects:


```python
e = {}
e['a_string'] = 3.3
e[3445] = 2.2
e[complex(2,1)] = 'value'
```


```python
print(e)
```


```python
e[3445]
```

If you try and access an element that does not exist, you will get a ``KeyError``:


```python
e[4]
```

Also, note that dictionaries do *not* know about order, so there is no 'first' or 'last' element.

It is easy to check if a specific key is in a dictionary, using the ``in`` operator:


```python
"a" in d
```


```python
"t" in d
```

Note that this also works for lists:


```python
3 in [1,2,3]
```




    True


# Booleans, Tuples, and Dictionaries

## Booleans

A ``boolean`` is one of the simplest Python types, and it can have two values: ``True`` and ``False`` (with uppercase ``T`` and ``F``):


```python
a = True
b = False
```

Booleans can be combined with logical operators to give other booleans:


```python
True and False
```


```python
True or False
```


```python
(False and (True or False)) or (False and True)
```

Standard comparison operators can also produce booleans:


```python
1 == 3
```


```python
1 != 3
```


```python
3 > 2
```


```python
3 <= 3.4
```

## Exercise 1

Write an expression that returns ``True`` if ``x`` is strictly greater than 3.4 and smaller or equal to 6.6, or if it is 2, and try changing ``x`` to see if it works:


```python
x = 3.7

# your solution here

```

## Tuples

Tuples are, like lists, a type of sequence, but they use round parentheses rather than square brackets:


```python
t = (1, 2, 3)
```

They can contain heterogeneous types like lists:


```python
t = (1, 2.3, 'spam')
```

and also support item access and slicing like lists:


```python
t[1]
```


```python
t[:2]
```

The main difference is that they are **immutable**, like strings:


```python
t[1] = 2
```

## Dictionaries

One of the data types that we have not talked about yet is called *dictionaries* (``dict``). If you think about what a 'real' dictionary is, it is a list of words, and for each word is a definition. Similarly, in Python, we can assign definitions (or 'values'), to words (or 'keywords').

Dictionaries are defined using curly brackets ``{}``:


```python
d = {'a':1, 'b':2, 'c':3}
```

Items are accessed using square brackets and the 'key':


```python
d['a']
```


```python
d['c']
```

Values can also be set this way:


```python
d['r'] = 2.2
```


```python
print(d)
```

The keywords don't have to be strings, they can be many (but not all) Python objects:


```python
e = {}
e['a_string'] = 3.3
e[3445] = 2.2
e[complex(2,1)] = 'value'
```


```python
print(e)
```


```python
e[3445]
```

If you try and access an element that does not exist, you will get a ``KeyError``:


```python
e[4]
```

Also, note that dictionaries do *not* know about order, so there is no 'first' or 'last' element.

It is easy to check if a specific key is in a dictionary, using the ``in`` operator:


```python
"a" in d
```


```python
"t" in d
```

Note that this also works for lists:


```python
3 in [1,2,3]
```




    True


# Control Flow

## ``if`` statements

The simplest form of control flow is the ``if`` statement, which executes a block of code only if a certain condition is true (and optionally executes code if it is *not* true. The basic syntax for an if-statement is the following:

    if condition:
        # do something
    elif condition:
        # do something else
    else:
        # do yet something else

Notice that there is no statement to end the if statement, and the
presence of a colon (``:``) after each control flow statement. Python relies
on **indentation and colons** to determine whether it is in a specific block of
code.

For example, in the following code:


```python
a = 1

if a == 1:
    print("a is 1, changing to 2")
    a = 2

print("finished")
```

The first print statement, and the ``a = 2`` statement only get executed if
``a`` is 1. On the other hand, ``print "finished"`` gets executed regardless,
once Python exits the if statement.

**Indentation is very important in Python, and the convention is to use four spaces (not tabs) for each level of indent.**

Back to the if-statements, the conditions in the statements can be anything that returns a boolean value. For example, ``a == 1``, ``b != 4``, and ``c <= 5`` are valid conditions because they return either ``True`` or ``False`` depending on whether the statements are true or not.

Standard comparisons can be used (``==`` for equal, ``!=`` for not equal, ``<=`` for less or equal, ``>=`` for greater or equal, ``<`` for less than, and ``>`` for greater than), as well as logical operators (``and``, ``or``, ``not``). Parentheses can be used to isolate different parts of conditions, to make clear in what order the comparisons should be executed, for example:

    if (a == 1 and b <= 3) or c > 3:
        # do something

More generally, any function or expression that ultimately returns ``True`` or ``False`` can be used.

In particular, you can use booleans themselves:


```python
a = 7
cond = a==1 or a>7
if cond:
    print ("hit")
else:
    print ("miss")
```

## ``for`` loops

Another common structure that is important for controling the flow of execution are loops. Loops can be used to execute a block of code multiple times. The most common type of loop is the ``for`` loop. In its most basic form, it
is straightforward:

    for value in iterable:
        # do things

The ``iterable`` can be any Python object that can be iterated over. This
includes lists or strings.


```python
for x in [3, 1.2, 'a']:
    print(x)
```


```python
for letter in 'hello':
    print(letter)
```

A common type of for loop is one where the value should go between two integers with a specific set size. To do this, we can use the ``range`` function. If given a single value, it will allow you to iterate from 0 to the value minus 1:


```python
for i in range(5):
    print(i)
```


```python
for i in range(3, 12):
    print(i)
```


```python
for i in range(2, 20, 2):  # the third entry specifies the "step size"
    print(i)
```

If you try iterating over a dictionary, it will iterate over the **keys** (not the values), in no specific order:


```python
d = {'a':1, 'b':2, 'c':3}
for key in d:
    print(key)
```

But you can easily get the value with:


```python
for key in d:
    print(key, d[key])
```

## Building programs

These different control flow structures can be combined to form programs. For example, the following program will print out a different message depending on whether the current number in the loop is less, equal to, or greater than 10:


```python
for value in [2, 55, 4, 5, 12, 8, 9, 22]:
    if value > 10:
        print("Value is greater than 10 (" + str(value) + ")")
    elif value == 10:
        print("Value is exactly 10")
    else:
        print("Value is less than 10 (" + str(value) + ")")
```

## Exiting or continuing a loop

There are two useful statements that can be called in a loop - ``break`` and ``continue``. When called, ``break`` will exit the loop it is currently in:


```python
for i in range(10):
    print(i)
    if i == 3:
        break
```

The other is ``continue``, which will ignore the rest of the loop and go straight to the next iteration:


```python
for i in range(10):
    if i == 2 or i == 8:
        continue
    print(i)
```

## ``while`` loops

Similarly to other programming languages, Python also provides a ``while`` loop which is similar to a ``for`` loop, but where the number of iterations is defined by a condition rather than an iterator:

    while condition:
        # do something

For example, in the following example:


```python
a = 1
while a < 10:
    print(a)
    a = a * 1.5
print("Once the while loop has completed, a has the value", a)
```

the loop is executed until ``a`` is equal to or exceeds 10.


# Functions

## Syntax

The syntax for a **function** is:
    
    def function_name(arguments):
        # code here
        return values

Functions are the **building blocks** of programs - think of them as basic units that are given a certain input an accomplish a certain task. Over time, you can build up more complex programs while preserving readability.

Similarly to ``if`` statements and ``for`` and ``while`` loops, indentation is very important because it shows where the function starts and ends.

**Note**: it is a common convention to always use lowercase names for functions.

A function can take multiple arguments...


```python
def add(a, b):
    return a + b

print(add(1,3))
print(add(1.,3.2))
print(add(4,3.))
```

... and can also return multiple values:


```python
def double_and_halve(value):
    return value * 2., value / 2.

print(double_and_halve(5.))
```

If multiple values are returned, you can store them in separate variables.


```python
d, h = double_and_halve(5.)
```


```python
print(d)
```


```python
print(h)
```

Functions can call other functions:


```python
def do_a():
    print("doing A")
    
def do_b():
    print("doing B")
    
def do_a_and_b():
    do_a()
    do_b()
```


```python
do_a_and_b()
```

**Figuring out the right functions is half the trick behind a good program**. A good function has a clear purpose that can, ideally, be described in one sentence of natural language.

Beginners typically err on the side of defining too few functions and writing monsters spanning a couple of screen pages.  That's a clear indication that you're doing it wrong.  A good function can be taken in (albeit perhaps not understood) at a clance.

## Optional Arguments

In addition to normal arguments, functions can take **optional** arguments that can default to a certain value. For example, in the following case:


```python
def say_hello(first_name, middle_name='', last_name=''):
    print("First name: " + first_name)
    if middle_name != '':
        print("Middle name: " + middle_name)
    if last_name != '':
        print("Last name: " + last_name)
```

we can call the function either with one argument:


```python
say_hello("Michael")
```

and we can also give one or both optional arguments (and the optional arguments can be given in any order):


```python
say_hello("Michael", last_name="Palin")
```


```python
say_hello("Michael", middle_name="Edward", last_name="Palin")
```


```python
say_hello("Michael", last_name="Palin", middle_name="Edward")
```

## Built-in functions

Some of you may have already noticed that there are a few functions that are defined by default in Python:


```python
x = [1,3,6,8,3]
```


```python
len(x)
```


```python
sum(x)
```


```python
int(1.2)
```

A [full list of built-in functions](http://docs.python.org/3/library/functions.html) is available from http://docs.python.org. Note that there are not *that* many - these are only the most common functions. Most functions are in fact kept inside **modules**, which we will cover later.
