"""CLI to generate baseline tarball."""

import click
import glob
import hashlib
import logging
import tarfile

from pathlib import Path


def _sha512(filepath: Path):
    sha512_hash = hashlib.sha512()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(4096)  # Read file in chunks
            if not chunk:
                break
            sha512_hash.update(chunk)
    return sha512_hash.hexdigest()


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


def validate_version(version: int, baseline_dir: Path) -> None:
    """Check the requested version.

    check against both the Readme & current_version files.
    """

    version_file = baseline_dir / "current_version"

    def _write_version():
        with open(version_file.resolve(), "w") as vf:
            vf.write(f"{version}")

    try:
        with open(version_file, "r") as f:
            current_version = int(f.readline().strip())
        if current_version != version:
            click.confirm(
                f"VERSION argument {version} doesn't match contents of {version_file}. Update?",
                abort=True,
            )
            _write_version()
    except FileNotFoundError:
        click.confirm(f"version file {version_file} not found. Write?", abort=True)
        _write_version()
    except ValueError:
        click.confirm(
            "version file is not valid. Overwrite with requested version?", abort=True
        )
        _write_version()

    readme = baseline_dir / "README.md"
    try:
        if not _grep_string(f"{version}:", readme):
            raise SystemExit(
                f"{readme} does not have an entry for version {version}. Please update the README."
            )
    except FileNotFoundError:
        raise SystemExit(f"{readme} does not exist")


@click.command
@click.argument("version", type=click.IntRange(min=0))
def main(version: int):
    """Makes the baselines tarball.

    VERSION is the version number for the generated tarball.
    """

    outname = f"kamayan_regression_baselines_v{version}.tgz"
    baseline_dir = Path(__file__).parent.resolve()
    out_file = baseline_dir / outname
    if out_file.exists():
        click.confirm(f"Output file {out_file} already exists. Overwrite?", abort=True)

    validate_version(version, baseline_dir)

    file_types = ["current_version", "README.md", "*.phdf", "*phdf.xdmf"]
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
    logging.info(f"SHA-512 hash: {_sha512(out_file)}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
