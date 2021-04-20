# SuperFit: The Cascade Successor to GolemFit

The main idea here is to make this easy to follow, and have as few things to hard-code in as possible

My main motivation is to improve upon the fit parameter, weighter, prior, and posterior implementation that was in GF.

All of those things are closely linked, and so all of them should be together in the same object. My plan is to define all the parameters in a json file; this json file will be loaded in, the parameter objects buit, and a dictionary-like container of parameters will be made (name->object).

That dictionary-like object will be used and shared by the weighters 

## What it does

Parametrize nuisance parameters. Fit a given flux (data or MC) to those nuisance parameters. Calculate a likelihood (LLH) for this.
Calculate the LLHs for multiple different fluxes.
Find the most likely one 

# Contributing:

I'll eventually try to set up a list of issues on a workboard that people can volunteer to do.

Some major style guidelines 
 - every function and class should have a docstring that explains what the thing does
 - If you have a choice between clear (but verbose) and concise (but obtuse), choose clear
 - No tabs; use four spaces
 - Nothing should be hard-coded. Use configuration files
 - Use descriptive names for variables. If the function of a variable is not immediately obvious fro

And other major rules
 - raise exceptions at unexpected situations. The exception message should provide enough detail to debug 
 - always check inputs in functions. Use `isinstance` and not `type(...)==[...]` 
 - Always test your code before committing 

## Some other suggestions

### Classes 

Be careful that class attributes are only modified when they are expected to. 
 - For internal use, use names like `self._value` 
 - Use getters and setters. For the getters use property decorators, ie 
```
def value(self):
    return self._value
```
this ensures things aren't accdentally changed. Similarly, for setters 
```
def set_value(self, value):
    if not isinstance(value, [dtype]:
        Logger.Fatal("...", ValueError)
    self._value = value 
```
so things don't get set to an unexpected value. This ensures we run into the problem exactly when it's relevant! 

### Dependencies

Try to avoid using bloated uncommon dependencies except where necessary. Numpy, scipy, and matplotlib should have everything we'll need! 
