# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

**Preferred:** Use GitHub's private vulnerability reporting for this repo:
[Report a vulnerability](../../security/advisories/new) (Security tab → "Report a vulnerability").

**Alternative:** Email sergiogabrielacosta7@gmail.com with the subject line
`SECURITY: face-shield`.

You should receive an acknowledgment within 5 business days.

## What to Include

To help triage quickly, please include:

- A clear description of the vulnerability and its potential impact
- Steps to reproduce, or a minimal proof-of-concept
- Affected file(s), endpoint(s), or component(s)
- Any relevant logs, screenshots, or request/response captures
- Your assessment of severity (if you have one)

## Scope

This covers the Python adversarial-engine code. It has no network-facing component — reports should focus on issues in local file/input handling.

This project has no live deployment. Reports are limited to the source code itself.

**Out of scope:**
- Vulnerabilities requiring physical access to a device
- Social engineering or phishing against the maintainer
- Denial-of-service testing (please report the theoretical issue instead of running it)
- Issues in third-party dependencies (report upstream, but let me know so I can track/patch)

## This Is a Personal Project

This is a solo-maintained personal/portfolio project, not a commercial product with an SLA.
I'll do my best to respond promptly and fix confirmed issues, but there's no guaranteed
response time or bug bounty. Credit will be given in the fix commit/changelog unless you
prefer to stay anonymous.
