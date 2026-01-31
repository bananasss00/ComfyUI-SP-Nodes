import logging
from comfy.comfy_types import IO

logger = logging.getLogger(__name__)


# A simple class to hold ANSI color codes for console output.
# This makes the code more readable than embedding raw escape codes.
class colors:
    YELLOW = '\033[93m' # For the prefix
    GREEN = '\033[92m'  # For the value
    ENDC = '\033[0m'    # To reset color back to default

class SP_DebugLogger:
    """
    A debug node that logs the string representation of any input to the console.
    It uses a user-defined prefix and color-codes the output for better visibility.
    The node passes through the original input and also outputs the log message as a string.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "any_input": (IO.ANY, {}),
                "prefix": ("STRING", {"default": "DEBUG:"}),
            },
        }

    # The node has two outputs:
    # 1. The original, untouched input data (IO.ANY).
    # 2. The generated log message as a clean string (STRING).
    RETURN_TYPES = (IO.ANY, "STRING",)
    
    # Define the names for the outputs for clarity in the UI.
    RETURN_NAMES = ("output", "log_message_string",)

    FUNCTION = "log_and_pass"
    CATEGORY = 'SP-Nodes'
    DESCRIPTION = "Logs the string representation of any input to the console with a colored prefix."

    def log_and_pass(self, any_input, prefix):
        """
        The main function of the node.
        It logs the message and then returns the necessary outputs.
        """
        # Convert the input data to its string representation for logging.
        # This works for tensors, models, strings, numbers, etc.
        value_as_string = str(any_input)

        # Create the log message without color tags for the string output.
        clean_log_message = f"{prefix} {value_as_string}"

        # Create the log message with ANSI color tags for the console.
        # This makes the debug output easy to spot in the terminal.
        colored_log_message = f"{colors.YELLOW}{prefix}{colors.ENDC} {colors.GREEN}{value_as_string}{colors.ENDC}"

        # Log the colored message to the console using the INFO level.
        logger.info(colored_log_message)

        # Return a tuple containing:
        # 1. The original, unmodified input, allowing this node to be non-disruptive.
        # 2. The clean log message as a string, which can be used by other nodes.
        return (any_input, clean_log_message,)
    
# To register the node with ComfyUI, you would typically add a mapping like this
# in the '__init__.py' file of your custom node package.
NODE_CLASS_MAPPINGS = {
   "SP_DebugLogger": SP_DebugLogger
}