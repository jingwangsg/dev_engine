from ray.job_submission import JobSubmissionClient
import uuid
from typing import List, Dict
import subprocess
import os
import os.path as osp


def zip_workdir(workdir: str, zip_path: str, exclude: List[str] = []):
    exclude_args = " ".join([f"--exclude {e}" for e in exclude])
    cmd = f"fd --extension py --extension sh --extension yaml --extension json {exclude_args} . | zip -@ {zip_path}"
    subprocess.run(cmd, shell=True, cwd=workdir)
    print(f"Zipped {workdir} to {zip_path}")


class JobSubmissionClientX:
    def __init__(self, *args, **kwargs):
        self.client = JobSubmissionClient(*args, **kwargs)

    def list_jobs(self):
        return self.client.list_jobs()

    def delete_job(self, job_id: str):
        return self.client.delete_job(job_id)

    def submit(
        self,
        entrypoint: str,
        runtime_env: dict = None,
        job_name: str = "submission",
        dev_python_packages: List[str] = [],
        working_dir_exclude: List[str] = [],
        when_exists: str = "raise",
        exists_status: List[str] = ["RUNNING", "SUCCEEDED"],
        **kwargs,
    ):
        random_id = str(uuid.uuid4()[:8])
        submission_id = f"{job_name}-{random_id}"

        # Override default values with environment variables
        when_exists = os.getenv("RAY_JOB_WHEN_EXISTS", "raise")
        if "RAY_JOB_EXISTS_STATUS" in os.environ:
            exists_status = os.getenv(
                "RAY_JOB_EXISTS_STATUS", "RUNNING,SUCCEEDED"
            ).split(",")

        # Deal with existing jobs
        existing_job_ids = [
            job.submission_id
            for job in self.list_jobs()
            if job.submission_id is not None and job.status.value in exists_status
        ]
        matched_job_ids = [
            job_id
            for job_id in existing_job_ids
            if job_id.rsplit("-", 1)[0] == job_name
        ]

        if len(matched_job_ids) > 0:
            if when_exists == "skip":
                print(
                    f"Job {job_name} already exists ({', '.join(matched_job_ids)}). Skipping submission."
                )
                return None
            elif when_exists == "raise":
                if submission_id in existing_job_ids:
                    input(
                        f"Job {submission_id} already exists. Press Enter to continue..."
                    )
            elif when_exists == "overwrite":
                pass
            else:
                raise ValueError(f"Invalid value for when_exists: {when_exists}")

        # Zip the working directory
        os.makedirs(osp.expanduser("~/.tmp/ray_workdir_cache"), exist_ok=True)

        zip_path = osp.expanduser(f"~/.tmp/ray_workdir_cache/{submission_id}.zip")
        zip_workdir(os.getcwd(), zip_path, exclude=working_dir_exclude)

        runtime_env["env_vars"] = runtime_env.get("env_vars", {}) | {
            "PYTHONPATH": ":".join(
                [osp.join(os.getcwd(), p) for p in dev_python_packages]
                + [os.getenv("PYTHONPATH", "")]
            )
        }

        # Submit the job
        self.client.submit_job(
            entrypoint=entrypoint,
            runtime_env={"working_dir": zip_path, **runtime_env},
            job_id=submission_id,
            **kwargs,
        )

        return submission_id


# class GearJobSubmissionClient:
#     def __init__(self, workflow_ids: List[str]):
#         self.

#     def submit(
#         self,
#         entrypoint: str,
#         runtime_env: dict = None,
#         job_name: str = "submission",
#         when_exists: str = "raise",
#         exists_status: List[str] = ["RUNNING", "SUCCEEDED"],
#         workflow_id: str = None,
#         **kwargs,
#     ):
#         pass
