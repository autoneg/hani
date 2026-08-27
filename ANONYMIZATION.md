# Anonymization notes (branch: `anonymized`)

This branch tracks `main` with author- and institution-identifying material removed,
for double-blind conference submission.

Note: this file deliberately contains no names, emails, URLs, or host names — writing
the audit terms down here would defeat its own purpose. The maintainer keeps the audit
term list outside the repository.

## What was removed / changed

**Deleted**
- `LICENSE` — names the copyright holder.
- `coding_agents/` — internal development notes containing real home-directory paths,
  usernames, a plaintext credential, and deployment host names.
- `AGENTS.md`, `CLAUDE.md` — maintainer-internal, first-person release instructions.
- `.github/workflows/{pypi,ontags,docs}.yml` — release/publish plumbing tied to the
  public package index and docs site.

**Scrubbed**
- `README.md` — package-index and docs badges, documentation-site link, public
  playground link, clone URL, author section; license section replaced with a
  placeholder.
- `pyproject.toml` — `authors` set to `Anonymous`; `[project.urls]` removed.
- `mkdocs.yml` — `site_author`, `site_url`, `repo_name`, `repo_url`, social links,
  `copyright`.
- `docs/` — author line, repository/source links, clone URL; install-from-index
  instructions replaced with install-from-source.
- `CHANGELOG.md` — repository / docs / issue-tracker links.
- `tests/test_support_agent.py` — a first name used as a test fixture value.
- The institutional deployment host name replaced with a placeholder across
  `src/hani/app.py`, `run.sh`, `runguest.sh`, `runreg.sh`, `docs/prolific.md`.

## How to hand this to reviewers

**Export the tree — do not clone or zip the repository.** Every commit on this branch
still carries the author's name and email in its git author/committer metadata, and the
`origin` remote points at the real repository.

```bash
git archive -o submission.zip anonymized
```

Then verify the exported contents against your own audit term list before sending:

```bash
git archive anonymized | tar -xO | grep -niE "<your audit terms>"
```

## Keeping this branch following `main`

Always merge `main` **into** `anonymized`, never the reverse — the other direction would
leak the scrub back into `main`:

```bash
git switch anonymized && git merge main
```

Re-run the audit over the tracked tree after each merge:

```bash
git ls-files -z | xargs -0 grep -rniE "<your audit terms>"
```

## Residual exposure this branch does NOT close

- The package and project name are published on the public package index under the
  author's name, so a reviewer who searches for them can find the authors. Renaming the
  package was out of scope; decide per venue whether it matters.
- Git history and commit metadata — see the export note above.
