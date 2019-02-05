Tests
=====
We are using ```pytest``` to run tests. 

Checklist
=============
Before running tests, make sure 
1. Your environment is properly setup
2. You build the latest code by running 
    
    ```python setup.py install```
   
   from the Ludwig root directory 
3. you have write access on the machine. Some of the tests
require saving data to disk

Running tests
=============
To run all tests, just run
```pytest``` from the ludwig root directory.

To run a single test, run
``` 
pytest path_to_filename::test_method_name
```

Example
-------
```
pytest tests/integration_tests/test_experiment.py::test_visual_question_answering
```


