.PHONY: run lint check test clean clean-models

# Set up environments (if needed) and start the server (HOST/PORT via run.sh).
run:
	./run.sh

# Auto-fix lint issues and format the code.
lint:
	uv run ruff check --fix
	uv run ruff format

# Read-only gate: lint + security (what CI runs, minus the tests).
check:
	uv run ruff check
	uv run bandit -q -c pyproject.toml -r server tools

# Run the platform-independent pipeline tests.
test:
	uv run pytest -q

# Remove generated jobs + audio (keeps the venvs and downloaded models).
clean:
	rm -rf data/jobs.sqlite data/jobs.sqlite-* data/audio/*.m4a

# Remove the virtual environments (model weights stay in the HF cache).
clean-models:
	rm -rf .venv xtts_engine/.venv
