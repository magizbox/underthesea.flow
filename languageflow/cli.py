# -*- coding: utf-8 -*-
import click
from languageflow.data_fetcher import DataFetcher


@click.group()
def main(args=None):
    """Console script for languageflow"""
    pass


@main.command()
@click.argument('dataset', required=True)
@click.argument('url', required=False)
def download(dataset, url):
    DataFetcher.download_dataset(dataset, url)


@main.command()
@click.option('-a', '--all', is_flag=True, required=False)
def list(all):
    DataFetcher.list(all)


@main.command()
@click.argument('data', required=True)
def remove(data):
    DataFetcher.remove(data)


if __name__ == "__main__":
    main()
