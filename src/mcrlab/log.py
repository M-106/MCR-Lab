# -----------
# > Imports <
# -----------
import logging



# -------------
# > Functions <
# -------------
def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%d.%m.%d %H:%M"
    )
    logger = logging.getLogger(__name__)

    # # supress others
    # logging.getLogger("datasets").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger



class LoggerPrinter:
    def info(self, msg):
        print(msg)



