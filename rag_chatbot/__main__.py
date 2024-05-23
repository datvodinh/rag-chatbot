import argparse
import llama_index
from dotenv import load_dotenv
from .ui import LocalChatbotUI
from .pipeline import LocalRAGPipeline
from .logger import Logger
from .ollama import run_ollama_server, is_port_open

load_dotenv()

# CONSTANTS
LOG_FILE = "logging.log"
DATA_DIR = "data/data"
AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]

# PARSER
parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="localhost",
    help="Set host to local or in docker container"
)
parser.add_argument(
    "--share", action='store_true',
    help="Share gradio app"
)
args = parser.parse_args()

# OLLAMA SERVER
if args.host != "host.docker.internal":
    port_number = 11434
    if not is_port_open(port_number):
        run_ollama_server()

# LOGGER

llama_index.core.set_global_handler("simple")
logger = Logger(LOG_FILE)
logger.reset_logs()

# PIPELINE
pipeline = LocalRAGPipeline(host=args.host)

# UI
ui = LocalChatbotUI(
    pipeline=pipeline,
    logger=logger,
    host=args.host,
    data_dir=DATA_DIR,
    avatar_images=AVATAR_IMAGES
)

ui.build().launch(
    share=args.share,
    server_name="0.0.0.0",
    debug=False,
    show_api=False
)
