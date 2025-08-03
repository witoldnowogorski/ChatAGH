# 🔧 Developer Environment Setup (UV + Streamlit + LangChain)

This project uses **[UV](https://github.com/astral-sh/uv)** as a dependency and environment manager
## Prerequisites

- **Python** (>= 3.10)
- **Pip**
- **UV** (installed globally)
- **Git**

### Install UV
If you don’t have UV yet:
```bash
pip install uv
```
or download the binary from the [official UV documentation](https://github.com/astral-sh/uv).

---
## Install dependencies

The project uses `pyproject.toml` and `uv.lock` to ensure consistent library versions.

```bash
uv sync
```

After running this command:
- UV will create and manage a virtual environment for the project.
- All required dependencies will be installed automatically.
- By default, UV uses an isolated environment in its own cache.

### Using `.venv` inside the project
If you prefer a `.venv` folder within the project directory (like traditional Python workflows), run:
```bash
uv sync --python .venv
```
This will create a `.venv` directory and install dependencies inside it.
### Adding a dependency
To add a new package (for example `requests`):
```bash
uv add requests
```

- This updates the `pyproject.toml` file with the new dependency.
- The `uv.lock` file is regenerated to include the exact version.


### Removing a dependency

To remove an existing package (for example `requests`):
```bash
uv remove requests
```

- The package is removed from the `pyproject.toml` file.
- The `uv.lock` file is updated accordingly.

---

## Run the application

This project uses **Streamlit** as its main application framework.

```bash
uv run streamlit run streamlit_app.py
```

The application will start and be available by default at `http://localhost:8501`.

---

## Environment configuration

The project requires an `.env` file with your environment variables.

A template file `.env.template` is provided.  
Create your own `.env` file:
```bash
cp .env.template .env
```
and fill in the required values.
