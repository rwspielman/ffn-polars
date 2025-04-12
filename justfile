test:
  uv run pytest

coverage:
    uv run pytest --cov --cov-report=term-missing

# build and install Rust plugin into venv
dev:
  maturin develop

# build wheel for PyPI (universal for Python version)
build:
  maturin build

# clean everything
clean:
  cargo clean
  rm -rf target
  rm -rf *.egg-info
  rm -rf dist
