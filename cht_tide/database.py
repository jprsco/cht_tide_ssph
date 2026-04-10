"""Tide model database management.

Provides :class:`TideModelDatabase` for discovering, caching, and
downloading tidal model datasets (e.g. FES2014) from a local directory
and/or an S3 bucket.
"""

import os

import boto3
import toml
from botocore import UNSIGNED
from botocore.client import Config

from cht_tide.fes2014 import TideModelFes2014


class TideModelDatabase:
    """Registry of available tide model datasets.

    Reads dataset metadata from a local ``tide_models.tml`` index file,
    optionally synchronising with an S3 bucket to discover new datasets.

    Parameters
    ----------
    path : str, optional
        Local directory where tide model datasets are stored.
    s3_bucket : str, optional
        S3 bucket name for online synchronisation.
    s3_key : str, optional
        S3 key prefix under which tide model data lives.
    s3_region : str, optional
        AWS region of the S3 bucket.
    check_online : bool, optional
        If ``True``, query S3 for new datasets on construction.
    """

    def __init__(
        self,
        path: str = None,
        s3_bucket: str = None,
        s3_key: str = None,
        s3_region: str = None,
        check_online: bool = False,
    ) -> None:
        self.path = path
        self.dataset = []
        self.s3_client = None
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.s3_region = s3_region
        self.read()
        if check_online:
            self.check_online_database()

    def read(self) -> None:
        """Read metadata for all datasets listed in ``tide_models.tml``.

        Skips datasets whose metadata file cannot be found. Populates
        ``self.dataset`` with model instances.
        """
        if self.path is None:
            print("Path to tide model database not set !")
            return

        # Check if the path exists. If not, create it.
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Read in database
        tml_file = os.path.join(self.path, "tide_models.tml")
        if not os.path.exists(tml_file):
            print(f"Warning! Tide model database file not found: {tml_file}")
            return

        datasets = toml.load(tml_file)

        for d in datasets["dataset"]:
            name = d["name"]

            if "path" in d:
                path = d["path"]
            else:
                path = os.path.join(self.path, name)

            # Read the meta data for this dataset
            fname = os.path.join(path, "metadata.tml")

            if os.path.exists(fname):
                metadata = toml.load(fname)
                dataset_format = metadata["format"]
            else:
                print(
                    f"Could not find metadata file for dataset {name} ! Skipping dataset."
                )
                continue

            if dataset_format.lower() == "fes2014":
                model = TideModelFes2014(name, path)
            elif dataset_format.lower() == "tpxo_old":
                pass

            self.dataset.append(model)

    def check_online_database(self) -> None:
        """Synchronise the local database with the S3 bucket.

        Downloads ``tide_models.tml`` from S3 and adds any new datasets
        (metadata only) to the local database, then re-reads the database.
        """
        if self.s3_client is None:
            self.s3_client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            )
        if self.s3_bucket is None:
            return
        key = f"{self.s3_key}/tide_models.tml"
        filename = os.path.join(self.path, "tide_models_s3.tml")
        print("Updating tide models database ...")
        try:
            self.s3_client.download_file(
                Bucket=self.s3_bucket,
                Key=key,
                Filename=filename,
            )
        except Exception:
            print(
                f"Failed to download {key} from {self.s3_bucket}. Database will not be updated."
            )
            return

        short_name_list, long_name_list = self.dataset_names()
        datasets_s3 = toml.load(filename)
        tide_models_added = False
        added_names = []
        for d in datasets_s3["dataset"]:
            s3_name = d["name"]
            if s3_name not in short_name_list:
                print(f"Adding tide model {s3_name} to local database ...")
                path = os.path.join(self.path, s3_name)
                os.makedirs(path, exist_ok=True)
                key = f"{self.s3_key}/{s3_name}/metadata.tml"
                filename = os.path.join(path, "metadata.tml")
                try:
                    self.s3_client.download_file(
                        Bucket=self.s3_bucket,
                        Key=key,
                        Filename=filename,
                    )
                except Exception as e:
                    print(e)
                    print(f"Failed to download {key}. Skipping tide model.")
                    continue
                tide_models_added = True
                added_names.append(s3_name)
        if tide_models_added:
            d = {}
            d["dataset"] = []
            for name in short_name_list:
                d["dataset"].append({"name": name})
            for name in added_names:
                d["dataset"].append({"name": name})
            with open(os.path.join(self.path, "tide_models.tml"), "w") as tml:
                toml.dump(d, tml)
            self.dataset = []
            self.read()

    def get_dataset(self, name: str):
        """Retrieve a dataset by its short name.

        Parameters
        ----------
        name : str
            Short name of the dataset.

        Returns
        -------
        TideModel or None
            The matching dataset, or ``None`` if not found.
        """
        for dataset in self.dataset:
            if dataset.name == name:
                return dataset
        return None

    def dataset_names(self) -> tuple:
        """Return lists of short and long dataset names.

        Returns
        -------
        tuple of (list of str, list of str)
            ``(short_name_list, long_name_list)``.
        """
        short_name_list = []
        long_name_list = []
        for dataset in self.dataset:
            short_name_list.append(dataset.name)
            long_name_list.append(dataset.long_name)
        return short_name_list, long_name_list
