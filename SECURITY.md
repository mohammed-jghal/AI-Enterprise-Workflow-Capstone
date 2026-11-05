# Security

This repository is a reference implementation and must be hardened before public deployment.

- Do not expose `/logs` without authentication and authorization.
- Treat model pickle files as trusted-only artifacts; never load an artifact from an untrusted upload.
- Put the API behind TLS, a reverse proxy, rate limiting, and network-level access controls.
- Keep secrets out of the repository and inject them through the deployment environment.
- Report security issues privately to the repository owner rather than opening a public issue.
