# Contributing to Fluid Shape Pipeline

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/chenyihao/fluid-shape-pipeline.git
cd fluid-shape-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use the following tools to maintain code quality:

- **black**: Code formatting (line length: 100)
- **isort**: Import sorting (black profile)
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:

```bash
black .
isort .
flake8 .
mypy sim extract ml vision tests
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=sim --cov=extract --cov=ml --cov=vision
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, semicolons, etc.)
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build process or auxiliary tool changes

Example:
```
feat: add ConvNeXt backbone support

- Add ConvNeXtEncoder class
- Update MultiScaleWakeNet to support ConvNeXt
- Add config option for backbone selection
```

## Pull Request Process

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes
3. Run tests and ensure they pass
4. Update documentation if needed
5. Submit a pull request

## Code Review

All submissions require review before merging. We aim to respond within 48 hours.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
