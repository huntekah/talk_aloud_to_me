.PHONY: run clean clean-models

# Set up environments (if needed) and start the server on 127.0.0.1:8765.
run:
	./run.sh

# Remove generated jobs + audio (keeps the venvs and downloaded models).
clean:
	rm -rf data/jobs.sqlite data/jobs.sqlite-* data/audio/*.m4a

# Remove the virtual environments (model weights stay in the HF cache).
clean-models:
	rm -rf .venv .venv-xtts
