#!/usr/bin/env python
"""Build the Sphinx docs for one version and deploy them with mike.

mike's CLI builds with MkDocs; its API exposes a context manager where the
caller performs the build, which is what this script uses for Sphinx::

    python docs/deploy_version.py 0.2 --title 0.2.1 --alias latest --set-default --push

The deployed versions (``versions.json`` on the docs branch, plus the version
being deployed) are materialised as ``docs/_versions.json`` *before* the build
so the version selector is rendered at build time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mike import git_utils
from mike.commands import AliasType, deploy, list_versions, set_default

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = DOCS_DIR / "_build" / "html"
VERSIONS_FILE = DOCS_DIR / "_versions.json"


def _sort_key(version: str) -> tuple:
    parts = version.split(".")
    if all(part.isdigit() for part in parts):
        return (1, [int(part) for part in parts])
    return (0, [version])


def _write_versions_file(version: str, title: str | None, aliases: list[str], branch: str) -> None:
    entries = {
        str(info.version): {
            "version": str(info.version),
            "title": info.title,
            "aliases": list(info.aliases),
        }
        for info in list_versions(branch)
    }
    entries[version] = {"version": version, "title": title or version, "aliases": list(aliases)}
    payload = sorted(entries.values(), key=lambda item: _sort_key(item["version"]), reverse=True)
    VERSIONS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"+ wrote {VERSIONS_FILE.relative_to(REPO_ROOT)} ({len(payload)} version(s))", flush=True)


def _has_default(branch: str) -> bool:
    """Whether the docs branch already has a root redirect (a default version)."""
    try:
        git_utils.read_file(branch, "index.html", universal_newlines=True)
    except git_utils.GitError:
        return False
    return True


def _sync_branch(branch: str, remote: str) -> None:
    """Point the local docs branch at the remote one so commits keep history.

    mike reuses an existing *local* branch; without this, a fresh CI clone
    (which only has ``<remote>/<branch>``) would make mike create an unrelated
    orphan branch and the push would be rejected as non-fast-forward.
    """
    remote_ref = f"{remote}/{branch}"
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", remote_ref],
        cwd=REPO_ROOT, capture_output=True,
    )
    if probe.returncode != 0:
        print(f"+ {remote_ref} not found; mike will create {branch} from scratch", flush=True)
        return
    subprocess.run(["git", "branch", "-f", branch, remote_ref], cwd=REPO_ROOT, check=True)
    print(f"+ {branch} -> {remote_ref}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="docs version to deploy, e.g. 0.2 or dev")
    parser.add_argument("--title", default=None, help="descriptive title (defaults to the version)")
    parser.add_argument("--alias", action="append", default=[],
                        help="alias pointing at this version (repeatable)")
    parser.add_argument("--branch", default="gh-pages", help="docs branch (default: gh-pages)")
    parser.add_argument("--remote", default="origin", help="remote to push to")
    parser.add_argument("--push", action="store_true", help="push the docs branch after committing")
    parser.add_argument("--set-default", action="store_true",
                        help="redirect the site root to this version")
    parser.add_argument("--set-default-if-missing", action="store_true",
                        help="redirect the site root only if no default exists yet")
    parser.add_argument("--dry-run", action="store_true",
                        help="build only (no branch commit, no push)")
    args = parser.parse_args()

    _write_versions_file(args.version, args.title, args.alias, args.branch)

    env = {
        **os.environ,
        "DOCS_VERSION": args.version,
        "MIKE_VERSIONS_FILE": str(VERSIONS_FILE),
    }
    build = [
        sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html",
        str(DOCS_DIR), str(SITE_DIR),
    ]

    if args.dry_run:
        subprocess.run(build, check=True, env=env, cwd=REPO_ROOT)
        return 0

    _sync_branch(args.branch, args.remote)

    cfg = {"site_dir": str(SITE_DIR), "use_directory_urls": True}
    # `copy` aliases: the deployed Pages artifact must work without symlinks.
    with deploy(cfg, args.version, title=args.title, aliases=args.alias,
                update_aliases=True, alias_type=AliasType.copy, branch=args.branch):
        subprocess.run(build, check=True, env=env, cwd=REPO_ROOT)

    if args.set_default or (args.set_default_if_missing and not _has_default(args.branch)):
        set_default(args.version, branch=args.branch)

    if args.push:
        command = ["git", "push", args.remote, args.branch]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT)

    print(f"+ deployed docs version {args.version} to {args.branch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())