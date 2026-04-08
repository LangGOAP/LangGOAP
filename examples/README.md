# LangGoap Notebooks

This directory contains Jupyter notebooks demonstrating LangGoap.

## Running Notebooks with Docker

```bash
cd examples
docker compose up
```

Open `http://127.0.0.1:8888/tree` in your browser.

When running with Docker Compose, the local library from `../` is mounted and installed automatically.

To stop:

```bash
docker compose down
```

## Running Notebooks Locally

```bash
pip install langgoap jupyter
jupyter notebook
```
