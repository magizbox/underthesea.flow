import os
import re
import shutil
from enum import Enum
from typing import Union, List, Dict

from flair.data import TaggedCorpus, Sentence, Token
from flair.data_fetcher import NLPTaskDataFetcher
from tabulate import tabulate

from languageflow.data import log
from languageflow.datasets import REPO
from languageflow.file_utils import cached_path, CACHE_ROOT
from pathlib import Path
import zipfile

MISS_URL_ERROR = "Caution:\n  With closed license dataset, you must provide URL to download"


class NLPData(Enum):
    VLSP2013_POS = "vlsp2013_pos"


class DataFetcher:

    @staticmethod
    def load_corpus(data: Union[NLPData, str], base_path: [str, Path] = None, url: str = None) -> TaggedCorpus:
        DataFetcher.download_dataset(data.name, url)

        if not base_path:
            base_path = Path(CACHE_ROOT) / 'datasets'
        data_folder = base_path / data.name.lower()
        if data == NLPData.VLSP2013_POS:
            columns = {0: 'text', 1: 'pos'}
            return DataFetcher.load_column_corpus(data_folder, columns)

    @staticmethod
    def load_column_corpus(
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        tag_to_biloes=None) -> TaggedCorpus:
        """
        Helper function to get a TaggedCorpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_biloes: whether to convert to BILOES tagging scheme
        :return: a TaggedCorpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith('.gz'): continue
                if 'train' in file_name and not '54019' in file_name:
                    train_file = file
                if 'dev' in file_name:
                    dev_file = file
                if 'testa' in file_name:
                    dev_file = file
                if 'testb' in file_name:
                    test_file = file

            # if no test file is found, take any file with 'test' in name
            if test_file is None:
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith('.gz'): continue
                    if 'test' in file_name:
                        test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train and test data
        sentences_train: List[Sentence] = DataFetcher.read_column_data(train_file, column_format)

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_column_data(test_file, column_format)
        else:
            sentences_test: List[Sentence] = [sentences_train[i] for i in
                                              DataFetcher.__sample(len(sentences_train), 0.1)]
            sentences_train = [x for x in sentences_train if x not in sentences_test]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            sentences_dev: List[Sentence] = DataFetcher.read_column_data(dev_file, column_format)
        else:
            sentences_dev: List[Sentence] = [sentences_train[i] for i in
                                             DataFetcher.__sample(len(sentences_train), 0.1)]
            sentences_train = [x for x in sentences_train if x not in sentences_dev]

        if tag_to_biloes is not None:
            # convert tag scheme to iobes
            for sentence in sentences_train + sentences_test + sentences_dev:
                sentence: Sentence = sentence
                sentence.convert_tag_scheme(tag_type=tag_to_biloes, target_scheme='iobes')

        return TaggedCorpus(sentences_train, sentences_dev, sentences_test, name=data_folder.name)

    @staticmethod
    def __sample(total_number_of_sentences: int, percentage: float = 0.1) -> List[int]:
        import random
        sample_size: int = round(total_number_of_sentences * percentage)
        sample = random.sample(range(1, total_number_of_sentences), sample_size)
        return sample

    @staticmethod
    def read_column_data(path_to_column_file: Path,
                         column_name_map: Dict[int, str],
                         infer_whitespace_after: bool = True):
        """
        Reads a file in column format and produces a list of Sentence with tokenlevel annotation as specified in the
        column_name_map. For instance, by passing "{0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}" as column_name_map you
        specify that the first column is the text (lexical value) of the token, the second the PoS tag, the third
        the chunk and the forth the NER tag.
        :param path_to_column_file: the path to the column file
        :param column_name_map: a map of column number to token annotation name
        :param infer_whitespace_after: if True, tries to infer whitespace_after field for Token
        :return: list of sentences
        """
        sentences: List[Sentence] = []

        try:
            lines: List[str] = open(str(path_to_column_file), encoding='utf-8').read().strip().split('\n')
        except:
            log.info('UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(path_to_column_file))
            lines: List[str] = open(str(path_to_column_file), encoding='latin1').read().strip().split('\n')

        # most data sets have the token text in the first column, if not, pass 'text' as column
        text_column: int = 0
        for column in column_name_map:
            if column_name_map[column] == 'text':
                text_column = column

        sentence: Sentence = Sentence()
        for line in lines:

            if line.startswith('#'):
                continue

            if line.strip().replace('﻿', '') == '':
                if len(sentence) > 0:
                    sentence.infer_space_after()
                    sentences.append(sentence)
                sentence: Sentence = Sentence()

            else:
                fields: List[str] = re.split("\t", line)
                token = Token(fields[text_column])
                for column in column_name_map:
                    if len(fields) > column:
                        if column != text_column:
                            token.add_tag(column_name_map[column], fields[column])

                sentence.add_token(token)

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            sentences.append(sentence)

        return sentences

    @staticmethod
    def download_dataset(data: str, url: str):
        if data not in REPO:
            print(f"No matching distribution found for '{data}'")
            return

        filepath = REPO[data]["filepath"]
        cache_dir = REPO[data]["cache_dir"]
        filepath = Path(CACHE_ROOT) / cache_dir / filepath
        if Path(filepath).exists():
            print(f"Data is already existed: '{data}' in {filepath}")
            return

        if data == "VNESES":
            url = "https://www.dropbox.com/s/m4agkrbjuvnq4el/VNESEcorpus.txt?dl=1"
            cached_path(url, cache_dir=cache_dir)
            shutil.move(Path(CACHE_ROOT) / cache_dir / "VNESEcorpus.txt?dl=1",
                        Path(CACHE_ROOT) / cache_dir / filepath)

        if data == "VNTQ_SMALL":
            url = "https://www.dropbox.com/s/b0z17fa8hm6u1rr/VNTQcorpus-small.txt?dl=1"
            cached_path(url, cache_dir=cache_dir)
            shutil.move(Path(CACHE_ROOT) / cache_dir / "VNTQcorpus-small.txt?dl=1",
                        Path(CACHE_ROOT) / cache_dir / filepath)

        if data == "VNTQ_BIG":
            url = "https://www.dropbox.com/s/t4z90vs3qhpq9wg/VNTQcorpus-big.txt?dl=1"
            cached_path(url, cache_dir=cache_dir)
            shutil.move(Path(CACHE_ROOT) / cache_dir / "VNTQcorpus-big.txt?dl=1",
                        Path(CACHE_ROOT) / cache_dir / filepath)

        if data == "VNTC":
            url = "https://www.dropbox.com/s/4iw3xtnkd74h3pj/VNTC.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VNTC.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2013-WTK":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2013-WTK.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2013_POS":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2013-POS.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VTB-CHUNK":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VTB-CHUNK.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2016-NER":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2016-NER.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2018-NER":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2018-NER.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

    @staticmethod
    def list(all):
        datasets = []
        for key in REPO:
            name = key
            type = REPO[key]["type"]
            license = REPO[key]["license"]
            year = REPO[key]["year"]
            directory = REPO[key]["cache_dir"]
            if not all:
                if license == "Close":
                    continue
            if license == "Close":
                license = "Close*"
            datasets.append([name, type, license, year, directory])

        print(tabulate(datasets,
                       headers=["Name", "Type", "License", "Year", "Directory"],
                       tablefmt='orgtbl'))

        if all:
            print(f"\n{MISS_URL_ERROR}")

    @staticmethod
    def remove(data):
        if data not in REPO:
            print(f"No matching distribution found for '{data}'")
            return
        dataset = REPO[data]
        cache_dir = Path(CACHE_ROOT) / dataset["cache_dir"]
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        print(f"Dataset {data} is removed.")
