from neo4j import GraphDatabase, Driver
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class GraphDBManager:
    def __init__(self, uri, user, password):
        self._driver: Driver = None
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            log.info("Successfully connected to Neo4j.")
            self.verify_connection()
        except Exception as e:
            log.error(f"Failed to connect to Neo4j: {e}")

    @property
    def driver(self) -> Driver:
        return self._driver

    def close(self):
        if self._driver is not None:
            self._driver.close()
            log.info("Neo4j connection closed.")

    def run_query(self, query, parameters=None):
        """
        Runs a single read query in an auto-commit transaction.
        """
        if self._driver is None:
            log.error("Driver not initialized. Cannot run query.")
            return []

        with self._driver.session() as session:
            try:
                result = session.run(query, parameters)
                return [record for record in result]
            except Exception as e:
                log.error(f"Error running query: {e}")
                return []

    def verify_connection(self):
        """
        Verifies the connection to the database is alive.
        """
        if self._driver is None:
            log.error("Driver not initialized.")
            return False
        try:
            self._driver.verify_connectivity()
            log.info("Neo4j connection is alive.")
            return True
        except Exception as e:
            log.error(f"Neo4j connection verification failed: {e}")
            return False
