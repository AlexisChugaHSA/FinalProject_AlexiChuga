import logging
import os

import pandas as pd
from sqlalchemy import create_engine

from benchmark import benchmark
from logger import configure_logging


def main():
    log = logging.getLogger(__name__)
    log.info("Starting...")

    # the data dir
    data_dir = os.path.join("..","..", "data")

    # the flight file
    csv_file = os.path.join(data_dir, "stats.csv")

    # the database file
    db_file = os.path.join(data_dir, "game_of_life.sqlite")
    # reading the csv file
    with benchmark(f"Reading {csv_file}", log):
        log.debug(f"Reading {csv_file} into a dataframe..")
        with open(csv_file, "r") as file:
            df = pd.read_csv(file)

            # the name of the columns into lowercase
            df.columns = [col.lower() for col in df.columns]
            log.debug("Reading done")

    # creating the sqlalchemy engine
    engine = create_engine(f"sqlite:///{db_file}")

    # insert the pandas dataframe into the database
    with benchmark(f"Inserting data into the database", log):
        with engine.begin() as connection:
            df.to_sql(
                "simulation_stats",
                con=connection,
                if_exists="replace",
                index=False)
    log.info("Done")


if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()