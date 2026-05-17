# Ludwig Codebase Review

Perform a thorough, opinionated code review of the Ludwig codebase (or a specified subsystem if an argument is given).

## Scope

If $ARGUMENTS is provided, scope the review to that subsystem or file pattern (e.g. `ludwig/features/`, `data pipeline`, `ray backend`).
Otherwise review the entire codebase.

## Review Dimensions

Evaluate each area across ALL of the following axes:

### Technical axes

- **Code smells**: long methods, god objects, feature envy, primitive obsession, data clumps, shotgun surgery, dead code
- **Duplication**: copy-paste logic, structural duplication, near-duplicate classes that should share a base
- **Abstraction level**: too low (leaking internals), too high (over-engineered), mismatched levels within a single function
- **Naming**: violate "naming things" rules — misleading names, abbreviations, overly generic names (`utils`, `helper`, `Manager`), names that lie about what a thing does, names that describe implementation not intent
- **Type hints**: missing, incomplete, `Any`-abuse, wrong (e.g. `dict` where `dict[str, float]` is knowable)
- **Docstrings**: missing on public API, wrong (describe what not why), stale (describe removed behavior)
- **Test coverage**: untested public surface, tests that only test the happy path, tests that mock away the thing being tested, missing edge cases
- **Performance**: unnecessary copies, redundant I/O, blocking the event loop, O(N²) in disguise, missing caching
- **Consistency**: same concept named differently in different files, different patterns for the same operation, inconsistent error handling styles

### Persona axes

Rate severity from each perspective and explain why it matters to that audience:

- **ML Engineer** (building production pipelines): Does this cause silent failures? Surprise OOMs? Hard-to-debug errors? Bad default choices?
- **ML Researcher** (running experiments): Is the config surface clear? Can they reproduce results? Do names match paper terminology? Is the API discoverable?
- **Open Source Contributor** (first PR): Is the code navigable? Is there a clear pattern to follow? Are there unexplained magic constants? Is test setup obvious?
- **Social Media ML Reader** (HN/Reddit/X): Would they call this "spaghetti"? Is there obvious NIH syndrome? Would they praise the architecture or cringe at it?

## Output Format

Structure the review as:

### Executive Summary

2-3 sentences on overall health and the single most important thing to fix.

### Critical Issues (must fix)

Numbered list. Each entry: file:line_range, what's wrong, why it matters, concrete fix.

### Major Issues (should fix)

Same format. Things that hurt quality but aren't blocking.

### Minor Issues (nice to fix)

Grouped by category (naming, type hints, docstrings, etc.).

### Persona Verdicts

One paragraph per persona with their honest take.

### Improvement Plan

Ordered list of PRs/tasks to address everything, with rough size estimate (S/M/L/XL).

## Instructions

- Be specific: always cite file paths and line numbers (or ranges)
- Be opinionated: don't hedge with "consider maybe possibly"
- Don't praise things that are merely adequate
- Distinguish between subjective style and objective bugs
- Focus on patterns, not one-off issues — if the same problem appears in 10 files, name the pattern once and give 3 examples
- Use the Explore subagent for broad searches, then Read for deep dives on critical files
