"""Run a user-installed FMM Docker image from the preprocessing pipeline."""

from __future__ import annotations

import logging
import os

import docker
from docker.errors import ImageNotFound


logger = logging.getLogger(__name__)


class FmmDocker:
    """Execute ``stmatch`` in an existing FMM Docker image.

    The repository does not build or distribute FMM. Users prepare the image
    separately; this class only starts a temporary container, mounts ``data/``,
    runs map matching, and removes the container.
    """

    def __init__(self, image_tag: str | None = None) -> None:
        self.image_tag = image_tag or os.environ.get(
            "FMM_DOCKER_IMAGE", "fmm:0.1.0"
        )
        self.data_full_path = os.path.abspath("data")
        self.mount_path = "/fmm/data"
        self.client = docker.from_env()
        try:
            self.client.images.get(self.image_tag)
        except ImageNotFound as exc:
            raise RuntimeError(
                f"FMM Docker image '{self.image_tag}' is not installed. "
                "Run `docker build -f utils/Dockerfile.fmm -t fmm:0.1.0 .` "
                "before preprocessing."
            ) from exc

    def __call__(
        self,
        processed_data_dir: str,
        edges_file: str,
        filtered_points_file: str,
    ) -> None:
        os.makedirs(processed_data_dir, exist_ok=True)
        container = self.client.containers.run(
            image=self.image_tag,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.data_full_path: {"bind": self.mount_path, "mode": "rw"}
            },
        )
        try:
            self._map_matching(
                container.id,
                processed_data_dir,
                edges_file,
                filtered_points_file,
            )
        finally:
            container.remove(force=True)
            logger.info("Removed temporary FMM container %s", container.short_id)

    def _container_path(self, host_path: str) -> str:
        absolute = os.path.abspath(host_path)
        relative = os.path.relpath(absolute, self.data_full_path)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            raise ValueError(
                f"FMM input must be inside {self.data_full_path}: {host_path}"
            )
        return f"{self.mount_path}/{relative.replace(os.sep, '/')}"

    def _map_matching(
        self,
        container_id: str,
        processed_data_dir: str,
        edges_file: str,
        filtered_points_file: str,
    ) -> None:
        output_file = os.path.join(processed_data_dir, "mr.txt")
        command = [
            "stmatch",
            "--network",
            self._container_path(edges_file),
            "--gps_point",
            "--gps",
            self._container_path(filtered_points_file),
            "--output",
            self._container_path(output_file),
            "--output_fields",
            "opath,cpath,speed",
            "--network_id",
            "fid",
            "--source",
            "u",
            "--target",
            "v",
            "-k",
            "4",
            "-r",
            "0.003",
            "-e",
            "0.0005",
            "--vmax",
            "0.0006",
            "--use_omp",
            "--step",
            "10000",
        ]
        logger.info("Run FMM map matching with image %s", self.image_tag)
        exec_id = self.client.api.exec_create(container_id, command)["Id"]
        for chunk in self.client.api.exec_start(exec_id, stream=True):
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                logger.info("%s", line)
        exit_code = self.client.api.exec_inspect(exec_id)["ExitCode"]
        if exit_code != 0:
            raise RuntimeError(f"FMM stmatch failed with exit code {exit_code}")
        logger.info("FMM map matching finished: %s", output_file)
