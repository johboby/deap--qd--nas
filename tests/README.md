# DEAP QD-NAS Test Suite

This directory contains the test suite for the DEAP QD-NAS framework.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_test_functions.py   # Test functions library tests
├── test_qd_nas.py          # QD-NAS core module tests
├── test_archive.py         # Archive manager tests
└── README.md               # This file
```

## Running Tests

### Prerequisites

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest tests/ -v
```

Or use the test runner script:
```bash
python run_tests.py
```

### Run Specific Tests

```bash
# Specific test file
pytest tests/test_test_functions.py -v

# Specific test class
pytest tests/test_test_functions.py::TestZDTFunctions -v

# Specific test method
pytest tests/test_test_functions.py::TestZDTFunctions::test_zdt1 -v

# Tests matching a pattern
pytest tests/ -k "zdt" -v
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# Generate terminal report with missing lines
pytest tests/ --cov=src --cov-report=term-missing

# Combine with specific test file
pytest tests/test_test_functions.py --cov=src.core.test_functions
```

## Writing Tests

### Basic Test Structure

```python
import pytest
import numpy as np
from src.core.test_functions import TestFunctionLibrary

class TestZDTFunctions:
    
    def test_zdt1(self):
        x = np.random.rand(10)
        f1, f2 = TestFunctionLibrary.zdt1(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))
```

### Using Fixtures

```python
# Using fixtures from conftest.py
def test_with_sample_data(sample_architecture):
    assert 'layers' in sample_architecture
    assert len(sample_architecture['layers']) > 0
```

### Testing Exceptions

```python
def test_invalid_function():
    with pytest.raises(ValueError):
        get_test_function('nonexistent_function')
```

### Parametrized Tests

```python
@pytest.mark.parametrize("func_name", ["zdt1", "zdt2", "zdt3"])
def test_zdt_functions(func_name):
    func = get_test_function(func_name)
    x = np.random.rand(10)
    result = func(x)
    assert len(result) == 2
```

## Test Coverage Goals

- Core modules: > 80%
- QD-NAS modules: > 70%
- Utility modules: > 60%

## Fixtures Available

- `random_seed` - Sets numpy random seed to 42
- `sample_data` - Dictionary with random x and y arrays
- `sample_architecture` - Sample neural architecture dict
- `sample_metrics` - Mock metrics object

## Debugging Tests

```bash
# Stop at first failure
pytest tests/ -x

# Run last failed tests only
pytest tests/ --lf

# Enter debugger on failure
pytest tests/ --pdb

# Show print statements
pytest tests/ -s

# Verbose output
pytest tests -vv
```

## Continuous Integration

Tests are designed to run in CI environments. Example GitHub Actions:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=src --cov-report=xml
```

## Best Practices

1. **Test Independence**: Each test should be able to run independently
2. **Clear Names**: Use descriptive test names that explain what is being tested
3. **Assert Specifically**: Use specific assertions rather than generic ones
4. **Test One Thing**: Each test should focus on a single behavior
5. **Use Fixtures**: Leverage fixtures for common setup code
6. **Keep It Simple**: Tests should be easy to read and understand

## Adding New Tests

When adding new features, always add corresponding tests:

1. Create test file in `tests/`
2. Name it `test_<module_name>.py`
3. Follow existing test patterns
4. Ensure good coverage of the new code
5. Update this README if adding new test categories

## Common Issues

### Import Errors

If you get import errors when running tests, ensure the project root is in your Python path:

```bash
export PYTHONPATH=/path/to/deap:$PYTHONPATH
pytest tests/
```

Or use the test runner script which handles this automatically.

### Random Test Failures

If tests fail randomly, it may be due to random number generation. The fixtures include a `random_seed` fixture to ensure reproducibility.

## Test Performance

To keep tests fast:
- Use small test data
- Avoid unnecessary computations
- Use mocks for slow operations
- Parametrize tests instead of duplicating code

## Questions or Issues

If tests are failing or you need help:
1. Check that all dependencies are installed
2. Ensure you're using the correct Python version
3. Run tests with `-v` for more detailed output
4. Check existing issues for similar problems
5. Create a new issue with test output and environment details
