# Contributing to LangGraph GroupChat

Thank you for your interest in contributing! We welcome bug fixes, features, documentation improvements, and examples.

## Quick Start

1. Fork and clone the repository
2. Install dependencies: `uv sync`
3. Create a feature branch: `git checkout -b feature-name`
4. Make your changes
5. Run tests and checks: `make test lint typecheck`
6. Submit a pull request

## Development Setup

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/yourusername/langgraph_groupchat_py.git
cd langgraph_groupchat_py
uv sync
source .venv/bin/activate  # Unix/macOS
```

**Available commands:**
- `make test` - Run tests
- `make lint` - Check code style
- `make format` - Auto-format code

## Coding Standards

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings (Google style) for public APIs
- Keep functions focused and well-named
- Maintain >80% test coverage

## Pull Request Guidelines

**Before submitting:**
- ✅ All tests pass
- ✅ Code is formatted and linted
- ✅ Type checking passes
- ✅ Documentation updated
- ✅ Tests added for new features

**PR checklist:**
- Clear, descriptive title
- Detailed description of changes
- Reference related issues (e.g., "Fixes #123")
- One PR per feature/fix
- Include tests and documentation

## Reporting Issues

**Bug reports should include:**
- Clear description and steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS)
- Minimal code example
- Full error traceback

**Feature requests should include:**
- Use case and motivation
- Proposed solution
- Alternative approaches considered

## Code of Conduct

Be respectful, constructive, and professional in all interactions.

## License

By contributing, you agree your contributions will be licensed under the project's license.

---

Questions? Open an issue or discussion. Thank you for contributing!
