# Agent Guidelines for negmas

## ⚠️ CRITICAL: GIT PUSH POLICY ⚠️

**NEVER NEVER NEVER NEVER PUSH WITHOUT EXPLICIT INSTRUCTION**

**ABSOLUTELY NO EXCEPTIONS. NEVER PUSH TO GITHUB WITHOUT THE USER EXPLICITLY SAYING "PUSH" OR GIVING A DIRECT COMMAND TO PUSH.**

**DO NOT PUSH EVEN IF:**
- All tests pass locally
- All tests pass on CI
- The changes look good
- You think it's ready
- You want to check CI

**ALWAYS:**
1. Run tests locally
2. Show results to user
3. **WAIT FOR EXPLICIT "PUSH" COMMAND**
4. Only then run `git push`



**Release Process:**

I do not use Travis CLI, I use Github Actions.

You must do the following steps for doing a release:

1. confirm the tests pass locally.
2. confirm that all commits since the last release are included in CHANGELOG.md (minor ones like chores may be ignored).
3. confirm the latest versions of negmas and negmas-llm dependencies on pypi and upgrade
   to them (clearly indicate that in your final report).
4. commit everything and push.
5. include any security prs that can be merged without breaking anything.
6. ASK ME AND CONFIRM the new version number except if I explicitly told you to release at
   a specific version (use that version) or to release unattended (choose an appropriate
   version yourself based on recent pypi version and pyproject.toml version). If you are going to ask me, give me the latest version on pypi and the version currently in pyproject.toml and suggest the new version.
7. tag with the new version number.
8. monitor Github actions until all actions pass
9. push with tags and monitor again.
10. after pypi is updated from github actions confirm the version.
11. make a release on github which includes the changes for this release from HISTORY.rst.
