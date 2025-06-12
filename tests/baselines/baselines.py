"""CLI to generate baseline tarball."""

import click
import glob
import hashlib
import json
import logging
import subprocess
import sys
import tarfile

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def _sha512(filepath: Path):
    sha512_hash = hashlib.sha512()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(4096)  # Read file in chunks
            if not chunk:
                break
            sha512_hash.update(chunk)
    return sha512_hash.hexdigest()


def _git_sha(path: Path) -> str:
    full_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    full_hash = full_hash.decode("utf-8").strip()
    return full_hash


def _grep_string(pattern: str, file: Path, case_sensitive: bool = False) -> bool:
    if not file.exists():
        raise FileNotFoundError

    with open(file, "r") as f:
        for line in f:
            search_pattern = pattern
            if not case_sensitive:
                line = line.lower()
                search_pattern = search_pattern.lower()

            if search_pattern in line:
                return True

    return False


@dataclass
class BaseLineVersion:
    """Relevant baseline version information.

    Attributes
        version: version number of baselines
        tar_sha: sha512 hash of the baselines tarball
            git_sha: git hash of repo at time of creation
        comment: comment describing changes to baselines
    """

    version: int
    git_sha: str
    tar_sha: str
    comment: str


class VersionHistory:
    """Class to hold the baseline version history."""

    def __init__(self, baseline_versions: None | Dict[int, BaseLineVersion] = None):
        self.baseline_versions: Dict[int, BaseLineVersion] = {}
        if baseline_versions:
            for bv in baseline_versions.values():
                self.add(
                    int(bv["version"]),
                    bv["git_sha"],
                    bv["tar_sha"],
                    bv["comment"],
                )

    def __getitem__(self, version: int) -> BaseLineVersion:
        return self.baseline_versions[version]

    @property
    def current_version(self) -> int:
        """Get the current version number."""
        return len(self.baseline_versions)

    def _add(self, new_version: BaseLineVersion):
        self.baseline_versions[new_version.version] = new_version

    def add(self, version: int, git_sha: str, tar_sha: str, comment: str):
        self._add(
            BaseLineVersion(
                version=version, git_sha=git_sha, tar_sha=tar_sha, comment=comment
            )
        )

    def contains(self, version: int) -> bool:
        return version in self.baseline_versions.keys()


def validate_version(version_file: Path) -> VersionHistory:
    """Check the requested version.

    check against both the Readme & current_version files.
    """

    try:
        with open(version_file, "r") as f:
            baseline_versions = VersionHistory(**json.loads(f.read()))
    except FileNotFoundError:
        raise FileNotFoundError("Version file 'baseline_versions.json' not found.")

    return baseline_versions


def _get_baseline_dir() -> Path:
    return Path(__file__).parent.resolve()


def _get_version_file() -> Path:
    return _get_baseline_dir() / "baseline_versions.json"


def _tarball_namer(version: int) -> str:
    return f"kamayan_regression_baselines_v{version}.tgz"


@click.group()
def cli():
    pass


@cli.command()
@click.argument("version", type=int)
def validate_tarball(version: int):
    """Validate the tarball sha512.

    VERSION the version number of the baseline tarball.
    """
    tar_sha = validate_version(_get_version_file())[version].tar_sha
    ball_sha = _sha512(_get_baseline_dir() / _tarball_namer(version))

    if tar_sha != ball_sha:
        sys.exit(
            f"tarball hash for {_tarball_namer(version)} doesn't match hash in {_get_version_file()}"
        )


@cli.command()
@click.option("--update", "-u", default=False)
def make_tarball(update: bool):
    """Makes the baselines tarball."""

    version_file = _get_version_file()

    try:
        baseline_versions = validate_version(version_file)
    except FileNotFoundError:
        click.confirm(
            "Version file 'baseline_versions.json' not found. Make new?", abort=True
        )
        baseline_versions = VersionHistory()

    version = baseline_versions.current_version
    if not update:
        version = click.prompt(
            f"Verion for tarbal, current:{version}, next:  ",
            default=version + 1,
            type=int,
        )
        update = baseline_versions.contains(version)

    outname = _tarball_namer(version)
    baseline_dir = _get_baseline_dir()
    out_file = baseline_dir / outname
    if out_file.exists():
        click.confirm(f"Output file {out_file} already exists. Overwrite?", abort=True)

    file_types = ["*.phdf", "*phdf.xdmf"]
    files_to_tar = []
    for ft in file_types:
        files_to_tar.extend(glob.glob(str(baseline_dir / ft)))

    with tarfile.open(str(out_file), "w:gz") as tar:
        logging.info(
            "Adding the following files to tarball:\n  " + "\n  ".join(file_types)
        )
        for file in files_to_tar:
            tar.add(file)

    logging.info(f"Creted tarball {out_file}.")
    tar_sha = _sha512(out_file)
    logging.info(f"SHA-512 hash: {tar_sha}.")
    git_sha = _git_sha(baseline_dir)

    comment_prompt = "Describe briefly why the baselines have been updated."
    default = None
    if update:
        default = baseline_versions[version].comment
        comment_prompt += f"\nCurrent comment:\n{default}\n"
    comment = click.prompt(comment_prompt, default=default)
    if comment is not None:
        baseline_versions.add(
            version=version, tar_sha=tar_sha, git_sha=git_sha, comment=comment
        )

    logging.info("Writng baseline_versions.json")
    with open(version_file, "w") as f:
        json.dump(baseline_versions, f, default=lambda o: o.__dict__, indent=2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    cli()
