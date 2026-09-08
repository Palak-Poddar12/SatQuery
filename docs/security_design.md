# SatQuery AI — Security Design

## 1. Overview

SatQuery AI is an agentic remote-sensing assistant that allows users to upload satellite imagery and interact with an AI system using natural-language queries.

Because uploaded imagery and user queries are untrusted inputs, the backend implements multiple security controls before data reaches the AI/agent layer.

---

# 2. Security Objectives

The primary security objectives are:

- Prevent unauthorized API access.
- Prevent malicious or invalid file uploads.
- Limit resource exhaustion caused by oversized requests.
- Protect stored imagery from filename-based attacks.
- Maintain file integrity using SHA-256.
- Provide auditable records of security-relevant events.
- Prevent API secrets from being written directly to logs.
- Provide a foundation for future JWT/SSO authentication.

---

# 3. Protected Assets

## 3.1 Satellite imagery

Uploaded imagery may contain sensitive geographic or operational information.

Protection:

- MIME validation
- Image verification
- File-size restrictions
- UUID-based filenames
- Controlled upload directory

## 3.2 User queries

Queries may contain sensitive operational information.

Protection:

- API authentication
- Rate limiting
- Audit logging

## 3.3 API credentials

The master API key provides access to protected endpoints.

Protection:

- API key authentication
- API-key fingerprinting in logs
- Secrets supplied through environment variables

## 3.4 Execution traces

Agent execution information can reveal:

- Task types
- Models used
- Processing decisions

Protection:

- Controlled API access
- Audit logging
- Future authorization controls

## 3.5 Audit logs

Audit logs contain security events and operational metadata.

Protection:

- Structured JSONL format
- No raw API key storage
- Restricted server-side storage

---

# 4. Threat Model

## Threat 1 — Unauthorized API access

An attacker may attempt to access upload or query endpoints without credentials.

Mitigation:

- `x-api-key` authentication
- HTTP 401 for missing or invalid credentials

---

## Threat 2 — Malicious file upload

An attacker may upload a non-image file while using an image extension.

Example:

`malware.exe` renamed to `malware.png`

Mitigation:

- Content-based MIME detection using libmagic
- Pillow image verification
- MIME allowlist
- Server-generated UUID filename

---

## Threat 3 — Oversized upload / resource exhaustion

An attacker may send very large files to consume server memory or storage.

Mitigation:

- Configurable maximum upload size
- Chunked upload reading
- HTTP 413 for oversized files
- Rate limiting

---

## Threat 4 — Filename/path traversal

An attacker may attempt filenames such as:

`../../important_file`

Mitigation:

- Original filename is never used as the stored filename.
- UUIDs are generated server-side.
- Files are stored inside the dedicated upload directory.

---

## Threat 5 — API abuse / DoS

An attacker may repeatedly call API endpoints.

Mitigation:

- SlowAPI rate limiting
- Configurable requests-per-minute limit
- HTTP 429 response

---

## Threat 6 — Secret leakage through logs

Logging the actual API key could expose credentials.

Mitigation:

- API keys are never written directly to audit logs.
- A short SHA-256 fingerprint is logged instead.

---

## Threat 7 — File tampering

Uploaded files may be modified after processing.

Mitigation:

- SHA-256 hash recorded during upload.
- Hash can later be used for integrity verification.

---

# 5. Authentication

The current MVP uses a master API key.

Configuration:

`MASTER_API_KEY`

Protected endpoints require:

`x-api-key`

Future versions can replace or extend this mechanism using:

- JWT
- OAuth 2.0
- SSO
- Role-Based Access Control (RBAC)

---

# 6. Rate Limiting

The API uses SlowAPI.

Default:

`60 requests/minute`

The value is configurable through:

`RATE_LIMIT_PER_MINUTE`

When the limit is exceeded:

`HTTP 429 Too Many Requests`

is returned.

---

# 7. Secure File Upload Pipeline

The upload pipeline is:

```text
Client
  |
  v
API Authentication
  |
  v
Rate Limiting
  |
  v
Size Validation
  |
  v
MIME Detection
  |
  v
Image Verification
  |
  v
SHA-256
  |
  v
UUID Filename
  |
  v
Secure Storage
  |
  v
Audit Log