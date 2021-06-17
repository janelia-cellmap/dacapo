import dacapo
import click
import logging


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        case_sensitive=False
    ),
    default="INFO"
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option(
    "-r",
    "--run-name",
    required=True,
    type=str,
    help="The NAME of the run to train."
)
def train(run_name):
    dacapo.train(run_name)


@cli.command()
@click.option(
    "-r",
    "--run",
    required=True,
    type=str,
    help="The name of the run to validate."
)
@click.option(
    "-i",
    "--iteration",
    required=True,
    type=int,
    help="The iteration at which to validate the run."
)
def validate(run_name, iteration):
    dacapo.validate(run_name, iteration)


@cli.command()
@click.option(
    "-r",
    "--run",
    required=True,
    type=str,
    help="The name of the run to predict."
)
@click.option(
    "-i",
    "--iteration",
    required=True,
    type=int,
    help="The iteration weights and parameters to use for prediction."
)
@click.option(
    "-r",
    "--dataset",
    required=True,
    type=str,
    help="The name of the dataset to predict on."
)
def predict(
        run_name,
        iteration,
        dataset_name):
    dacapo.predict(run_name, iteration, dataset_name)
