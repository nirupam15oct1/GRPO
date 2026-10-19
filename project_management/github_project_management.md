Short answer: yes — git/GitHub has a full project-management layer, and you can tie it directly to specs. Let me ground the spec part with a quick check, since the tooling there is recent.

Yes, both are not only possible but standard practice. Here's how it actually works.

**The project-management layer in GitHub**

GitHub gives you four primitives that together act as a lightweight project manager, no extra tool needed:

- **Issues** — one per unit of work (a feature, a bug, a task). This is your to-do list. Each has a title, description, labels, assignee.
- **Projects** — a Kanban/table board that sits on top of your issues. Columns like *Backlog → In progress → Done*, or custom fields like priority, target date, which showcase it belongs to. You drag issue cards across columns.
- **Milestones** — a bucket of issues with a due date (e.g. "capability_showcase v1"). Shows % complete automatically as issues close.
- **Branches + Pull Requests** — the code side. You make a branch per issue, do the work, open a PR. The magic word is in the PR description: write `Closes #42` and when that PR merges, issue #42 auto-closes and its board card moves to Done.

So the whole chain is: *issue → branch → PR → merge → issue closes → board updates*. Git is doing the tracking for you as a side effect of normal work.

**A concrete flow for one of your projects**

1. In the `quantization-implementation` repo, create issues: "Add subagent orchestration," "Write quant eval harness," etc.
2. Make a Project board, drop those issues in, set a milestone with a date.
3. For each issue, branch (`git switch -c quant-eval-harness`), code, commit, push, open PR with `Closes #N`.
4. Merge → board self-updates. At any point the milestone tells you how much is left.

**Linking with spec**

Two levels here.

*Manual version (works today, zero setup):* keep a `spec.md` (or `/specs/` folder) in the repo. Each GitHub issue references the spec section it implements — "implements §3.2 of spec.md." The spec is version-controlled alongside code, so when the spec changes, that's a commit too, and you can see exactly which code change corresponds to which spec revision via git history. Issues become the bridge between spec and implementation.

*Tooled version — GitHub Spec Kit:* this is the current purpose-built tool for exactly what you're describing, and it's built for your stack — it works with Claude Code and installs via uv. GitHub open-sourced it (MIT-licensed) to move AI coding from ad-hoc prompting toward a repeatable, verifiable workflow. The idea: you write the specification first, then a technical plan, then a set of small testable tasks that the AI agent implements. At its heart is a "constitution" — a document capturing the non-negotiable principles for the project (for you, that maps naturally onto BFSI governance/compliance constraints).

Install it the way you already install things:
```
uvx --from git+https://github.com/github/spec-kit.git specify init <project>
```
It drops a `/specs/` structure and slash commands (`/specify`, plan, tasks, implement) into your repo that Claude Code drives. The spec lives in git, so it's versioned, diffable, and reviewable like code.

One honest caveat: GitHub itself frames Spec Kit as an experiment to test how well spec-driven development works, not a finished product — so treat it as scaffolding to adapt, not gospel.

Given you're running many parallel showcase repos through Claude Code, the combination that fits you best: **one Project board across repos** for the bird's-eye view + **Spec Kit per repo** so each project's spec drives its own implementation. Want me to sketch the board structure for your current set of showcase projects, or walk through setting up Spec Kit on one of them?

Sources:
- [GitHub Spec Kit (official repo)](https://github.com/github/spec-kit)
- [GitHub Open Sources Kit for Spec-Driven AI Development](https://visualstudiomagazine.com/articles/2025/09/03/github-open-sources-kit-for-spec-driven-ai-development.aspx)
- [GitHub Spec Kit: How It Works and When to Use It](https://codemyspec.com/blog/github-spec-kit-guide)
