# Contributing to DEAP QD-NAS

Thank you for your interest in contributing to the DEAP QD-NAS project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [issue tracker](https://github.com/your-username/deap-qdnas/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)

### Contributing Code

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub
git clone https://github.com/your-username/deap-qdnas.git
cd deap-qdnas
git remote add upstream https://github.com/original-owner/deap-qdnas.git
```

#### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

#### 3. Set Up Development Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install in development mode
pip install -e .
```

#### 4. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add or update tests as needed
- Update documentation if applicable

#### 5. Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run linting (if configured)
flake8 src/
black src/ --check
```

#### 6. Submit Pull Request

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to any related issues
- Screenshots/examples if applicable

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Add docstrings for public functions and classes

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Use pytest for testing
- Follow existing test patterns

### Documentation

- Update relevant documentation
- Add docstrings for new functions/classes
- Keep examples up to date
- Use clear, concise language

### Commit Messages

Use clear, descriptive commit messages. Consider using conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `perf:` Performance improvements

Example:
```
feat: add CVT-MAP-Elites algorithm

Implements CVT-MAP-Elites for improved space coverage.
Includes tests and documentation updates.

Closes #123
```

## Project Structure

```
src/
├── core/          # Core framework
├── nas/           # QD-NAS modules
├── algorithms/    # Algorithm implementations
├── advanced/      # Advanced features
├── utils/         # Utilities
└── applications/  # Applications

tests/             # Test suite
examples/          # Examples
docs/              # Documentation
```

## Adding New Features

### New Algorithm

1. Create implementation in appropriate module
2. Follow existing algorithm patterns
3. Add comprehensive tests
4. Update documentation
5. Add example usage

### New Test Function

1. Add to `src/core/test_functions.py`
2. Include reference to original paper
3. Add tests
4. Update documentation

### New QD-NAS Feature

1. Follow existing patterns in `src/nas/`
2. Ensure compatibility with existing modules
3. Add performance considerations
4. Include comprehensive tests

## Code Review Process

1. All PRs require review before merging
2. Address reviewer feedback promptly
3. Keep PRs focused on single feature/fix
4. Update PR based on feedback
5. Ensure CI checks pass

## Questions?

If you have questions:
- Check existing documentation
- Look through existing issues
- Create a new issue with your question
- Contact maintainers

Thank you for contributing!
