# -*- coding: utf-8 -*-
import click
from languageflow.data_fetcher import DataFetcher


@click.group()
def main(args=None):
    """Console script for languageflow"""
    pass


@main.command()
@click.argument('data', required=False)
@click.option('-f', '--force', is_flag=True, required=False)
def download(data, force):
    if data == None:
        DataFetcher.list()
    else:
        DataFetcher.download_data(data)
    # download_component(component, force)


if __name__ == "__main__":
    main()
