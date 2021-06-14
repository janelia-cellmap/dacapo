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
    "--run-id",
    required=True,
    type=str,
    help="The ID of the run to train."
)
def train(run_id):
    dacapo.train(run_id)


@cli.command()
@click.option(
    "-r",
    "--run-id",
    required=True,
    type=str,
    help="The ID of the run to validate."
)
@click.option(
    "-i",
    "--iteration",
    required=True,
    type=int,
    help="The iteration at which to validate the run."
)
def validate(run_id, iteration):
    dacapo.validate(run_id, iteration)


@cli.command()
@click.option(
    "-r",
    "--run-id",
    required=True,
    type=str,
    help="The ID of the run to predict."
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
    "--dataset-id",
    required=True,
    type=str,
    help="The ID of the dataset to predict on."
)
def predict(
        run_id,
        iteration,
        dataset_id):
    dacapo.predict(run_id, iteration, dataset_id)
