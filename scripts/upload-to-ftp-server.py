"""
Upload data to the FTP server
"""

from __future__ import annotations

import ftplib
import os.path
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import tqdm
import tqdm.utils
import typer
from loguru import logger

DEFAULT_LOGGING_CONFIG = dict(
    handlers=[
        dict(
            sink=sys.stderr,
            colorize=True,
            format=" - ".join(
                [
                    "<green>{time:!UTC}</>",
                    "<lvl>{level}</>",
                    "<cyan>{name}:{file}:{line}</>",
                    "<lvl>{message}</>",
                ]
            ),
        )
    ],
)
"""Default configuration used with :meth:`loguru.logger.configure`"""


def setup_logging(config: dict[str, Any] | None = None) -> None:
    """
    Early setup for logging.

    Parameters
    ----------
    config
        Passed to :meth:`loguru.logger.configure`. If not passed,
        :const:`DEFAULT_LOGGING_CONFIG` is used.
    """
    if config is None:
        config = DEFAULT_LOGGING_CONFIG

    logger.configure(**config)
    logger.enable("fgen")


@contextmanager
def login_to_ftp(ftp_server: str, username: str, password: str) -> ftplib.FTP:
    """
    Login to the FTP server
    """
    ftp = ftplib.FTP(ftp_server, passwd=password, user=username)  # noqa: S321
    logger.info(f"Logged into {ftp_server} using {username=} and {password=}")

    yield ftp

    ftp.quit()
    logger.info(f"Closed connection to {ftp_server}")


def cd_v(dir_to_move_to: str, ftp: ftplib.FTP) -> None:
    """
    Change directory, verbosely
    """
    ftp.cwd(dir_to_move_to)
    logger.debug(f"Now in {ftp.pwd()} on FTP server")


def mkdir_v(dir_to_make: Path, ftp: ftplib.FTP) -> None:
    """
    Make directory, don't fail if the directory already exists
    """
    try:
        logger.debug(f"Attempting to make {dir_to_make} on {ftp.host=}")
        ftp.mkd(dir_to_make)
        logger.debug(f"Made {dir_to_make} on {ftp.host=}")
    except ftplib.error_perm:
        logger.debug(f"{dir_to_make} already exists on {ftp.host=}")


def upload_file(file: Path, root_dir: Path, ftp: ftplib.FTP) -> None:
    """
    Upload file to the FTP server

    We ensure that the FTP connection exists
    in the same directory as it was on entry,
    irrespective of the directory creation and selection commands we run in this function.

    Parameters
    ----------
    file
        File to upload.
        The full path of the file relative to ``root_dir`` will be uploaded.
        In other words, any directories in ``file`` will be made on the
        FTP server before uploading.

    root_dir
        Root directory.
        Directories above ``root_dir`` will not be included in the upload path.

    ftp
        FTP connection to use for the upload.
    """
    logger.info(f"Uploading {file}")

    ftp_pwd_in = ftp.pwd()
    file_rel_to_root = file.relative_to(root_dir)
    logger.info(
        f"Relative to {ftp_pwd_in} on the FTP server, will upload to {file_rel_to_root}"
    )

    for parent in file_rel_to_root.parents[::-1]:
        if parent == Path("."):
            continue

        to_make = parent.parts[-1]
        mkdir_v(to_make, ftp=ftp)
        ftp.cwd(to_make)
        logger.debug(f"Now in {ftp.pwd()} on FTP server")

    logger.info(f"Uploading {file}")
    file_size = os.path.getsize(file)
    with open(file, "rb") as fh:
        upload_command = f"STOR {file.name}"
        logger.debug(f"Upload command: {upload_command}")

        try:
            with tqdm.tqdm(
                desc="Uploading",
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as tbar:
                reader_wrapper = tqdm.utils.CallbackIOWrapper(tbar.update, fh, "read")
                ftp.storbinary(upload_command, reader_wrapper)

            logger.info(f"Successfully uploaded {file}")
        except ftplib.error_perm:
            logger.exception(
                f"{file.name} already exists on the server in {ftp.pwd()}. "
                "Use a different root directory on the receiving server "
                "if you really wish to upload."
            )

    cd_v(ftp_pwd_in, ftp=ftp)


def upload_files(
    files_to_upload: Iterable[Path],
    root_dir_files_to_upload: str,
    ftp: ftplib.FTP,
    root_dir_ftp_incoming_files: str,
    dir_ftp_files_to_upload: str,
) -> None:
    """
    Upload files to the FTP server

    Parameters
    ----------
    files_to_upload
        Files to upload

    root_dir_files_to_upload
        Root directory of the files to upload.
        Directories above this directory will not be included
        in the uploaded files' paths.

    ftp
        FTP server connection.

    root_dir_ftp_incoming_files
        Root directory on the FTP server for receiving files.

    dir_ftp_files_to_upload
        Directory on the FTP server in which to upload files.
        This directory is always made (if it doesn't exist)
        within ``root_dir_ftp_incoming_files``.
    """
    cd_v(f"/{root_dir_ftp_incoming_files}", ftp=ftp)
    mkdir_v(dir_ftp_files_to_upload, ftp=ftp)

    cd_v(dir_ftp_files_to_upload, ftp=ftp)
    for file in files_to_upload:
        upload_file(file, root_dir=root_dir_files_to_upload, ftp=ftp)


def main(  # noqa: PLR0913
    files_to_upload: list[str],
    password: str,
    root_dir_files_to_upload_esgf_drs: str,
    ftp_server: str = "ftp.llnl.gov",
    username: str = "anonymous",
    root_dir_ftp_incoming_files: str = "incoming",
    dir_ftp_files_to_upload: str = "cr-ghgs-znicholls",
) -> None:
    """
    Upload to FTP server

    Parameters
    ----------
    files_to_upload
        Files to upload

    password
        Password to use when logging in.
        LLNL asks that you use your email here.

    root_dir_files_to_upload_esgf_drs
        Name of the root directory of the ESGF data reference syntax (DRS).
        This allows us to strip unneeded directories from the file paths
        when uploading.

    ftp_server
        FTP server to login to.

    username
        Username to use when logging in.

    root_dir_ftp_incoming_files
        Root directory for receiving files on the FTP server.

    dir_ftp_files_to_upload
        Directory in which to upload our files on the FTP server.
        This directory is always made relative to ``root_dir_ftp_incoming_files``.
    """
    with login_to_ftp(
        ftp_server=ftp_server, username=username, password=password
    ) as ftp_logged_in:
        upload_files(
            files_to_upload=[Path(v) for v in files_to_upload],
            root_dir_files_to_upload=root_dir_files_to_upload_esgf_drs,
            ftp=ftp_logged_in,
            root_dir_ftp_incoming_files=root_dir_ftp_incoming_files,
            dir_ftp_files_to_upload=dir_ftp_files_to_upload,
        )


if __name__ == "__main__":
    typer.run(main)
