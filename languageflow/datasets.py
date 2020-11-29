import os
import shutil
import zipfile
from pathlib import Path
from languageflow.file_utils import cached_path, CACHE_ROOT

REPO = {
    "AIVIVN2019_SA": {
        "type": "Categorized",
        "license": "Close",
        "year": "2019",
        "filepath": "",
        "tmp_file": "AIVIVN2019_SA.zip?dl=1"
    },
    "VLSP2018_NER": {
        "type": "Tagged",
        "license": "Close",
        "year": "2018",
        "filepath": "",
        "tmp_file": "VLSP2018-NER.zip?dl=1"
    },
    "VLSP2018_SA": {
        "cache_dir": "datasets/vlsp2018_sa",
        "type": "Categorized",
        "license": "Close",
        "year": "2018",
        "filepath": "",
        "tmp_file": "VLSP2018_SA.zip?dl=1"
    },
    "VTB_CHUNK": {
        "type": "Tagged",
        "license": "Close",
        "year": "2017",
        "filepath": "",
        "tmp_file": "VTB-CHUNK.zip?dl=1"
    },
    "UTS2017_BANK": {
        "type": "Categorized",
        "license": "Open",
        "year": "2017",
        "filepath": "",
        "url": "https://www.dropbox.com/s/xl8sof2i1c35n62/UTS2017_BANK.zip?dl=1",
        "tmp_file": "UTS2017_BANK.zip?dl=1"
    },
    "VLSP2016_NER": {
        "type": "Tagged",
        "license": "Close",
        "year": "2016",
        "filepath": "",
        "tmp_file": "VLSP2016-NER.zip?dl=1"
    },
    "VLSP2016_SA": {
        "cache_dir": "datasets/vlsp2016_sa",
        "type": "Categorized",
        "license": "Close",
        "year": "2016",
        "filepath": "",
        "tmp_file": "VLSP2016_SA.zip?dl=1"
    },
    "VLSP2013_WTK": {
        "cache_dir": "datasets/VLSP2013-WTK",
        "type": "Tagged",
        "license": "Close",
        "year": "2013",
        "filepath": "",
        "tmp_file": "VLSP2013-WTK.zip?dl=1"
    },
    "VLSP2013_POS": {
        "cache_dir": "datasets/VLSP2013-POS",
        "type": "Tagged",
        "license": "Close",
        "year": "2013",
        "filepath": "",
        "tmp_file": "VLSP2013-POS.zip?dl=1"
    },
    "VNESES": {
        "type": "Plaintext",
        "license": "Open",
        "year": "2012",
        "filepath": "VNESEScorpus.txt",
        "tmp_file": "VNESEcorpus.txt?dl=1",
        "url": "https://www.dropbox.com/s/m4agkrbjuvnq4el/VNESEcorpus.txt?dl=1"
    },
    "VNTQ_BIG": {
        "type": "Plaintext",
        "license": "Open",
        "year": "2012",
        "filepath": "VNTQcorpus-big.txt",
        "url": "https://www.dropbox.com/s/t4z90vs3qhpq9wg/VNTQcorpus-big.txt?dl=1",
        "tmp_file": "VNTQcorpus-big.txt?dl=1"
    },
    "VNTQ_SMALL": {
        "type": "Plaintext",
        "license": "Open",
        "year": "2012",
        "filepath": "VNTQcorpus-small.txt",
        "url": "https://www.dropbox.com/s/b0z17fa8hm6u1rr/VNTQcorpus-small.txt?dl=1",
        "tmp_file": "VNTQcorpus-small.txt?dl=1"
    },
    "VNTC": {
        "cache_dir": "datasets/VNTC",
        "type": "Categorized",
        "license": "Open",
        "year": "2007",
        "filepath": "",
        "url":"https://www.dropbox.com/s/4iw3xtnkd74h3pj/VNTC.zip?dl=1",
        "tmp_file": "VNTC.zip?dl=1"
    }
}


class Dataset:

    @staticmethod
    def get(id):
        if not Dataset.is_available(id):
            print(f"No matching distribution found for '{data}'")
            return None
        data = REPO[id]
        name = id
        type = data["type"]
        year = data["year"]
        license = data["license"]
        url = data["url"] if "url" in data else ""
        filepath = data["filepath"] if "filepath" in data else ""
        tmp_file = data["tmp_file"]
        dataset = Dataset(name=name, type=type, year=year, url=url, tmp_file=tmp_file, filepath=filepath, license=license)
        return dataset

    @staticmethod
    def is_available(id):
        return id in REPO

    def __init__(self, name=None, type=None, license=None, year=None, filepath="", url="", tmp_file=""):
        self.name = name
        self.type = type
        self.license = license
        self.year = year
        self.filepath = filepath
        self.url = url
        self.tmp_file = tmp_file
        self.cache_dir = Path(CACHE_ROOT) / "datasets" / self.name

    def exists(self):
        filepath = Path(CACHE_ROOT) / "datasets" / self.name
        return Path(filepath).exists()

    def info(self):
        content = {"name": self.name}
        return content

    def download(self):
        cached_path(self.url, cache_dir=self.cache_dir)
        if self.filepath == "":
            zip = zipfile.ZipFile(self.cache_dir / self.tmp_file)
            zip.extractall(self.cache_dir)
            os.remove(self.cache_dir / self.tmp_file)
        else:
            shutil.move(self.cache_dir / self.tmp_file, self.cache_dir / self.filepath)

