import logging
import os

from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker

from logger import configure_logging

Base = declarative_base()

class GameOfLife(Base):
    __tablename__ = "simulation_stats"

    id = Column(Integer, primary_key=True)
    iteration = Column(Integer, nullable=False)
    live_cells = Column(Integer, nullable=False)
    dead_cells = Column(Integer, nullable=False)
    duration_ms = Column(Float, nullable=False)


# @typechecked
def main():
    log = logging.getLogger(__name__)
    log.info("Starting...")

    # the data dir
    data_dir = os.path.join("..","..", "data")

    # the database file
    db_file = os.path.join(data_dir, "game_of_life.sqlite")

    # creating the sqlalchemy engine
    engine = create_engine(f"sqlite:///{db_file}")

    # create the database
    log.debug("Creating the database...")
    Base.metadata.create_all(engine)
    log.debug("... database created.")

    log.info("Done")

if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()