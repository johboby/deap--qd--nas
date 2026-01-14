import pytest
import numpy as np
from src.core.test_functions import (
    TestFunctionLibrary,
    ZDT_FUNCTIONS,
    DTLZ_FUNCTIONS,
    CONSTRAINED_FUNCTIONS,
    SINGLE_OBJECTIVE_FUNCTIONS,
    get_test_function,
    list_test_functions
)


class TestZDTFunctions:
    
    def test_zdt1(self):
        x = np.random.rand(10)
        f1, f2 = TestFunctionLibrary.zdt1(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))

    def test_zdt2(self):
        x = np.random.rand(10)
        f1, f2 = TestFunctionLibrary.zdt2(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))

    def test_zdt3(self):
        x = np.random.rand(10)
        f1, f2 = TestFunctionLibrary.zdt3(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))

    def test_zdt4(self):
        x = np.random.rand(10)
        f1, f2 = TestFunctionLibrary.zdt4(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))

    def test_zdt6(self):
        x = np.random.rand(10)
        f1, f2 = TestFunctionLibrary.zdt6(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))


class TestDTLZFunctions:
    
    def test_dtlz1(self):
        x = np.random.rand(10)
        objectives = TestFunctionLibrary.dtlz1(x, n_obj=2)
        assert len(objectives) == 2
        assert all(isinstance(obj, (int, float)) for obj in objectives)

    def test_dtlz2(self):
        x = np.random.rand(10)
        objectives = TestFunctionLibrary.dtlz2(x, n_obj=2)
        assert len(objectives) == 2
        assert all(isinstance(obj, (int, float)) for obj in objectives)

    def test_dtlz3(self):
        x = np.random.rand(10)
        objectives = TestFunctionLibrary.dtlz3(x, n_obj=2)
        assert len(objectives) == 2
        assert all(isinstance(obj, (int, float)) for obj in objectives)

    def test_dtlz4(self):
        x = np.random.rand(10)
        objectives = TestFunctionLibrary.dtlz4(x, n_obj=2)
        assert len(objectives) == 2
        assert all(isinstance(obj, (int, float)) for obj in objectives)

    def test_dtlz7(self):
        x = np.random.rand(10)
        objectives = TestFunctionLibrary.dtlz7(x, n_obj=2)
        assert len(objectives) == 2
        assert all(isinstance(obj, (int, float)) for obj in objectives)


class TestConstrainedFunctions:
    
    def test_constrained_zdt1(self):
        x = np.array([0.5, 0.5])
        (f1, f2), constraints = TestFunctionLibrary.constrained_zdt1(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))
        assert len(constraints) == 2

    def test_srn_constrained(self):
        x = np.array([2.0, 2.0])
        (f1, f2), constraints = TestFunctionLibrary.srn_constrained(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))
        assert len(constraints) == 2

    def test_bnh_constrained(self):
        x = np.array([3.0, 3.0])
        (f1, f2), constraints = TestFunctionLibrary.bnh_constrained(x)
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))
        assert len(constraints) == 2


class TestSingleObjectiveFunctions:
    
    def test_sphere(self):
        x = np.array([1.0, 2.0, 3.0])
        result = TestFunctionLibrary.sphere(x)
        assert isinstance(result, (int, float))
        assert result > 0

    def test_rastrigin(self):
        x = np.array([1.0, 2.0, 3.0])
        result = TestFunctionLibrary.rastrigin(x)
        assert isinstance(result, (int, float))

    def test_rosenbrock(self):
        x = np.array([1.0, 2.0, 3.0])
        result = TestFunctionLibrary.rosenbrock(x)
        assert isinstance(result, (int, float))

    def test_ackley(self):
        x = np.array([0.0, 0.0])
        result = TestFunctionLibrary.ackley(x)
        assert isinstance(result, (int, float))

    def test_griewank(self):
        x = np.array([0.0, 0.0])
        result = TestFunctionLibrary.griewank(x)
        assert isinstance(result, (int, float))


class TestFunctionUtilities:
    
    def test_get_test_function(self):
        func = get_test_function('zdt1')
        assert callable(func)
        f1, f2 = func([0.5, 0.5, 0.5])
        assert isinstance(f1, (int, float))
        assert isinstance(f2, (int, float))

    def test_get_test_function_invalid(self):
        with pytest.raises(ValueError):
            get_test_function('invalid_function')

    def test_list_test_functions_all(self):
        functions = list_test_functions()
        assert len(functions) > 0
        assert 'zdt1' in functions
        assert 'dtlz1' in functions

    def test_list_test_functions_by_category(self):
        zdt_functions = list_test_functions('zdt')
        assert 'zdt1' in zdt_functions
        assert 'zdt2' in zdt_functions

        dtlz_functions = list_test_functions('dtlz')
        assert 'dtlz1' in dtlz_functions
        assert 'dtlz2' in dtlz_functions

    def test_list_test_functions_invalid_category(self):
        with pytest.raises(ValueError):
            list_test_functions('invalid_category')


class TestFunctionCollections:
    
    def test_zdt_functions(self):
        assert 'zdt1' in ZDT_FUNCTIONS
        assert 'zdt2' in ZDT_FUNCTIONS
        assert callable(ZDT_FUNCTIONS['zdt1'])

    def test_dtlz_functions(self):
        assert 'dtlz1' in DTLZ_FUNCTIONS
        assert 'dtlz2' in DTLZ_FUNCTIONS
        assert callable(DTLZ_FUNCTIONS['dtlz1'])

    def test_constrained_functions(self):
        assert 'constrained_zdt1' in CONSTRAINED_FUNCTIONS
        assert 'srn_constrained' in CONSTRAINED_FUNCTIONS

    def test_single_objective_functions(self):
        assert 'sphere' in SINGLE_OBJECTIVE_FUNCTIONS
        assert 'rastrigin' in SINGLE_OBJECTIVE_FUNCTIONS
        assert callable(SINGLE_OBJECTIVE_FUNCTIONS['sphere'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
