# NuIsanceFit: The Python Successor to GolemFit

This software fits a set of nuisance parameters (see: [1](https://arxiv.org/abs/1909.01530)) according to a dataset and a set of MC simulation generated according to known parameters.
It is the spiritual successor to [GolemFit](https://github.com/icecube/GolemFit.)

<p align="center">
    <img src="https://user-images.githubusercontent.com/52141176/116719997-4f49d680-a9a1-11eb-9bc9-b550b13ba44f.png" width="350" height="350">
</p>
 
Motivations: 
 - Posterior, Fit Parameter, and Priors should be implemented in a refined way such that more can be added in a procedural way. 
 - Code should be easy to follow
 - New features should be easy to implement
 - Stability! Ill-configured parameters should be caught on launch, not after running for a while
 - Keep configuration separate from implementation. Users should only have to edit the configuration files, not the actual code. 

Ultimately this should be as easy to use as possible! 

## What it does

Parametrize nuisance parameters. 
Find the suite of nuisance parameters, that reweight simulation data, maximizing the likelihood of measuring some flux given that reweighted sim data. 
Calculate a likelihood (LLH) for this set of nuisance parameters according to their prior distributions and the difference between the simulation expectation and flux measurement. 
Calculate the LLHs for multiple different fluxes (different physics?)
Find the most likely one 

# Installing

## **PYTHON 2 Will Never Be Supported**

## Dependencies 

### Python 

You need to use Python 3. 
The rest of these are also now (or will soon be) necessary.
 - numpy
 - scipy
 - pytorch 
 - matplotlib 

### C++ dependencies:
 - Photospline (with python bindings)
 - SQuIDS (for nuSQuIDS)
 - nuSQuIDS and nuSQuIDSpy
 - LeptonWeighter (with python bindings) 

When in installing the squids and LW, remember to set your `LD_LIBRARY_PATH`.
Forgetting to do this can lead to frustrating symbol errors with mangled names. 

## Cython 

Some parts of this code are written in *cython*.
This is done to optimize the time it takes to run some of the more crunchy parts.
As a consequence, there's a short little installation process before you can actually use the code. 
First, set up the environment in your bashrc or bash profile (your call)
```
export PYTHONPATH=$PYTHONPATH:/path/to/here
```
Next, install cython.
The process will depend on your package manager; with apt it's as easy as 
```
sudo apt-get install cython3
```
Then, navigate to `/path/to/here/NuIsanceFit/cython_files/`, and run the bash script `cython_build.sh`. 
It will call `python3` and tell it to prepare the cython.


# Contributing:

Check out the workboard in the Projects tab! As I do more work here, I'll keep adding more projects there. 

## Major style guidelines 
 - functions and classes should have a docstring that explains what the thing does
 - If you have a choice between clear (but verbose) and concise (but obtuse), choose clear
 - No tabs; use four spaces
 - Nothing should be hard-coded. Use configuration files
 - Use descriptive names for variables. If the function of a variable is not immediately obvious, you should probably rename it. 

## And other major rules
 - raise exceptions at unexpected situations. The exception message should provide enough detail to debug 
 - always check inputs in functions. Use `isinstance` and not `type(...)==[...]` 
 - Always test your code before committing 
 - Do **not** import anything into the global namespace. No `from numpy import *` allowed. This is a pain for debugging, especially when there's overlap 


## Naming Schemes

I'm not the most consistent about this (yet). But, I'm working on switching to... 
 - use `camelCase` for the most part
 - Classes should start capitalized: `PowerLawTiltWeighter`
 - Prefix internal attributes and functions names with a `_`

## Assorted Suggestions

### Methods/Functions

Type-check inputs to methods and functions. 
Throw exceptions if an unexpected datatype is received. 

Also check against non-physical values. 

### Classes 

Be careful that class attributes are only modified when they are expected to. This can make for some hard to track down bugs!
 - Use getters and setters. For the getters use property decorators, ie 
```
@property
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

Also, use inheritance when appropriate (see: `weighter.py`) 


