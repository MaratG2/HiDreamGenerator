# Contributing

Start with the mock setup in README.md; no model download is needed for API or frontend work.

Before submitting a change:

- Run `python -m unittest discover -s tests -v` at the root, then `npm ci` and `npm run build` in `frontend`.
- For UI changes, exercise the affected flow at desktop and mobile sizes.
- For inference changes, document checkpoint, model type, seed, dimensions, scheduler and steps tested. API mocks alone do not validate model behavior.
- Keep credentials, `.env`, virtual environments, model weights and private outputs out of commits. Preserve upstream attribution.

Bug reports should include reproduction steps, expected/actual behavior, platform, GPU/VRAM when relevant, dependency versions and sanitized logs. Avoid sharing private prompts, images or access tokens.

The server owns one GPU pipeline per process. Multiple workers, persistent queues, authentication and shared hosting require explicit architecture work and tests for resource ownership and access boundaries.
