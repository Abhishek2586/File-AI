# Security Checklist
## AI File Assistant — Prompt 28

---

## Input Validation

- [x] File upload restricted to `.pdf` type only (Streamlit `type=["pdf"]` restriction)
- [x] File names are never directly executed — only read via PyMuPDF in a sandboxed manner
- [x] User chat input is passed as a string to the Q&A pipeline; no eval/exec is ever called
- [ ] Add max file size check (recommend: reject files > 50 MB)
- [ ] Sanitize filenames when saving to disk to avoid path traversal (e.g., `../../../etc/passwd.pdf`)

## API Key Security

- [x] API key stored in `.env` file, never hard-coded in source
- [x] `.env` is listed in `.gitignore` to prevent accidental commits
- [x] `.env.example` provided as a template (with placeholder values only)
- [x] Streamlit secrets support: `.streamlit/secrets.toml` can be used for cloud deployment
- [ ] Rotate the FastRouter API key if it has been accidentally committed to Git history

## Rate Limiting

- [x] `OpenAIHandler` uses exponential backoff retry logic on API failures
- [x] Embedding pipeline uses batch processing to avoid flooding the API
- [ ] Consider adding a Streamlit-level rate limiter (e.g., limit 1 upload per 60s per session)

## Data Privacy

- [x] PDF content stays local — files are processed in a temp directory and never uploaded to a third party
- [x] ChromaDB data is persisted locally in `data/chroma`
- [x] API calls only send text chunks (not entire files) to the LLM provider
- [ ] Implement explicit data retention policy: delete indexed data older than N days

## Session Security

- [x] No user accounts or passwords (reduces attack surface for single-user local deployments)
- [x] Chat history stored in Streamlit session state (in-memory, clears on restart)
- [ ] For multi-user deployments: add session isolation

## Logging

- [x] Python `logging` module is used throughout all backend modules
- [x] API call counts and times are logged to console
- [ ] Add security-event logging (e.g., log every file upload with timestamp)
- [ ] Redirect logs to a rotating file handler for production

## Dependency Scanning

- [ ] Run `pip install safety && safety check -r requirements.txt` to scan for known CVEs
- [ ] Run `pip install bandit && bandit -r modules/` for static code analysis
- [ ] Pin all dependency versions in `requirements.txt` to prevent supply-chain attacks

## HTTPS / Deployment

- [ ] Configure Streamlit to run behind a reverse proxy (nginx/Caddy) with HTTPS for production
- [ ] For Streamlit Cloud: HTTPS is automatically handled

---

## Priority Actions

| Priority | Action | Effort |
|----------|--------|--------|
| 🔴 High | Sanitize filenames before saving to temp dir | Low |
| 🔴 High | Run `safety check` to scan dependencies | Low |
| 🟡 Medium | Add max file size validation | Low |
| 🟡 Medium | Session timeout for long-running sessions | Medium |
| 🟢 Low | Security event logging | Medium |
