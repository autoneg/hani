# HANI Agent Development Guide

## Build & Test Commands
- **Install dependencies**: `uv sync --dev`
- **Run application**: `hani` or `python src/hani/run.py`
- **Run tests**: `pytest` (runs all tests)
- **Run single test**: `pytest path/to/test_file.py::test_function_name`
- **Type checking**: `mypy src/hani` (configured in pyproject.toml)

## Code Style Guidelines

### Imports
- Standard library first, then third-party, then local imports (separated by blank lines)
- Use absolute imports from `hani` package: `from hani.scenarios.base import ScenarioMaker`
- Group related imports: `from negmas import Negotiator, SAOMechanism, SAOState`

### Type Hints & Types
- **Required**: All function signatures must include type hints (params and return types)
- Use modern syntax: `list[str]`, `dict[str, Any]`, `tuple[float, Outcome]`
- Use `Protocol` for structural typing: `class ScenarioMaker(Protocol):`
- Use `@overload` for multiple type signatures (see `helpers/negotiators.py`)
- Nullable types: `type | None` (not `Optional[type]`)

### Code Structure
- Use `@define` from `attrs` for data classes (see `AppConfig`, `ToolConfig`)
- Use `Enum` for constants: `class Timing(Enum): Always = 0`
- Prefer composition over inheritance

### Naming Conventions
- Classes: `PascalCase` (e.g., `SAOHumanNegotiator`)
- Functions/variables: `snake_case` (e.g., `make_mechanism`, `human_index`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `SELECTED_AGENT_TYPES`)
- Private: prefix with `_` (e.g., `_parse`)

### Error Handling
- Use explicit exception handling with specific types
- Print traceback for debugging: `print(traceback.format_exc())`
- Validate inputs early: `raise ValueError(f"Cannot load scenario from {path}")`
