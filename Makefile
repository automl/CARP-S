.PHONY: test docs check

test:
	python -m pytest tests

docs:
	python -m webbrowser -t "http://127.0.0.1:8000/"
	python -m mkdocs serve --clean

check:
	pre-commit run --all-files