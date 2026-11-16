.PHONY: install install-frozen format check-format check-ruff lint test test-all clean check-types check docs docs-clean docs-serve test-coverage coverage-report coverage-html find-dead-code benchmark benchmark-compare

install:
	uv sync

install-frozen:
	uv sync --frozen

format:
	uv run isort ./langgoap ./tests/ --profile black
	uv run black ./langgoap ./tests/

check-format:
	uv run black --check ./langgoap ./tests/
	uv run isort ./langgoap ./tests/ --check-only --profile black

check-types:
	uv run python -m mypy ./langgoap

# Undefined names (the #71 NameError slipped through), unused imports and
# redefinitions.  black/isort do not catch these.
check-ruff:
	uv run ruff check --select F821,F401,F811 ./langgoap ./tests/

# lint CHECKS; it never rewrites files.  Use `make format` to fix formatting.
lint: check-format check-types check-ruff

test:
	uv run python -m pytest -vv -s --log-level=CRITICAL -m "not api"

test-all:
	uv run python -m pytest -vv -s --log-level=CRITICAL

test-coverage:
	uv run python -m pytest --cov=langgoap --cov-report=html --cov-report=term-missing --cov-report=xml --log-level=CRITICAL

coverage-report:
	uv run python -m coverage report

coverage-html:
	uv run python -m coverage html

find-dead-code:
	uv run python -m vulture langgoap --sort-by-size

benchmark:
	uv run pytest tests/benchmarks/ -v \
		--benchmark-sort=mean \
		--benchmark-columns=mean,stddev,min,max,rounds \
		--benchmark-group-by=func \
		--no-header \
		-p no:randomly

benchmark-compare:
	uv run pytest tests/benchmarks/ -v \
		--benchmark-sort=mean \
		--benchmark-columns=mean,stddev,min,max,rounds \
		--benchmark-compare \
		--benchmark-compare-fail=mean:10% \
		-p no:randomly

check: lint find-dead-code test

docs:
	uv run --group docs python docs/copy_notebooks.py
	uv run --group docs python docs/generate_api_index.py
	uv run --group docs sphinx-build -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build docs/api/generated
	rm -rf docs/examples/basics docs/examples/tutorials docs/examples/screencast

docs-serve: docs
	python -m http.server 8085 --directory docs/_build/html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "_build" -exec rm -rf {} +
