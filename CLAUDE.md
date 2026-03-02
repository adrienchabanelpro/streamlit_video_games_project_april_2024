# Claude Code Configuration

## Behavioral Rules (Always Enforced)

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- NEVER save working files, text/mds, or tests to the root folder
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files

## File Organization

- NEVER save to root folder — use the directories below
- Use `source/` for app source code
- Use `source/pages/` for Streamlit page modules
- Use `source/ml/` for ML inference code
- Use `scripts/` for training, data collection, and utility scripts
- Use `tests/` for test files
- Use `data/` for datasets
- Use `models/` for trained model artifacts
- Use `reports/` for training logs and evaluation outputs

## Project Architecture

- Keep files under 500 lines
- Use type hints on all new/modified functions
- Ensure input validation at system boundaries

## Build & Test

```bash
# Run the app
make run

# Test
make test

# Lint
make lint

# Format
make format

# Train models
make train
```

- ALWAYS run tests after making code changes
- ALWAYS verify the app starts before committing

## Security Rules

- NEVER hardcode API keys, secrets, or credentials in source files
- NEVER commit .env files or any file containing secrets
- Always validate user input at system boundaries
- Always sanitize file paths to prevent directory traversal
