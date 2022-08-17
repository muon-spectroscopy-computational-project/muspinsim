# How MuSpinSim Handles Input

Parameters are given using a specially formatted input file
  - A tab-formatted file,
  - comprised of a set of keywords, and indented set of values for those keywords.

Only specific keywords are allowed
  - which determine simulation properties/setup

Keywords have one or more values associated with them

  - These values can made up of one or more expressions or functions
    - we use a parser library `lark` to parse these inputs.

  - Some keywords allow specifying variables, and these can be used when specifying values for other keywords
    - see fitting variables keyword.


## Reading the file - input.py

We read the file and create an object `MuSpinInput()`
  - Passing the input file IO stream to the constructor:

  - This object holds a dictionary for each keyword and variable found

  `self._keywords = keyword_name: corresponding subclass of MuSpinKeyword()`

  `self._variables = variable_name: corresponding instantiation of FittingVariable() class`

1. Convert file into a dictionary of input blocks

  `keyword_name: [list of inputs]`

  - remove comments
  - if any blocks are improperly spaced/formatted, produce errors

2. Parse fitting keywords first

  - we want to find all fitting variables defined before parsing the rest of the keywords - as they might be used later
  - store any variables found - `self._variables`


3. Get Experiment Keyword if specified

  - if an experiment type is specified, we must create new keywords corresponding to the experiment type

4. Parse the rest of the keywords

  - try to create the corresponding `MuSpinKeyword()` object for the keyword
  - store keywords found - `self._keywords`

## Reading in Keywords Individually - keyword.py

Once we get the keyword name and a set of input values, we create an object to validate and store the inputs and keyword args.

`MuSpinKeyword` - base class
`MuSpinEvaluateKeyword` - base class for keywords which allow expressions/functions
  - uses parser to evaluate each input

`MuSpinExpandKeyword` - base class for keywords which allow range() function

`MuSpinCouplingKeyowrd` - base class for coupling interaction Keywords

Keyword object takes a list of inputs and a list of args
  - Each keyword has a set of allowed


Each keyword has different object - which specify:
  - keyword_name
  - block_size - number of inputs allowed
  - default - the default value used, if no value given
  - accept_range - if the keyword should accept range
  - _validators - a dictionary of
      `err_msg:<lambda function to determine validity>`
  - _functions - a dictionary of
      `function_name: <code to execute function>`
  - expr_size_bounds - tuple representing min and max number of space-separated elements to each input


## Parsing with lark - larkeval.py

Parsing is done when `self._store_values` is called for any
 `MuSpinEvaluateKeyword` object, or when the `evaluate()` function is called

When `self._store_values` is called - we create `LarkExpression` object
  - which will create a parse tree for the input in the constructor class

When `self.evaluate(**params)` - the parse tree is evaluated by calling
`evaluate` method for the LarkExpression object

## Catching and using variables - input.py/variables.py

Fitting variables are parsed first - provided fitting data is available. Each variable found will create a `Variable()` object which stores min-max bound + starting value

They are collected into `self._variables` - which is passed to any `MuSpinEvaluateKeyword` object.
  - if fitting variables are found when parsing, no error will show


## Get values after parsing - input.py

Once the input file is read in, and the `MuSpinInput()`` object is fully instantiated - with `self._variables` dictionary fully populated

We can run `evaluate()` method - This will create a dictionary where each keyword will be stored alongside its parsed values and args.




## Things to improve

Error handling limited to the individual keyword and values

- we should do some further checks here on whether  keywords and values are compatible with each other
