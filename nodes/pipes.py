
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
    
# ---------------------------------------------------------------------------
# BASE CLASS: Contains all the logic for any number of ports.
# This class is not registered directly with ComfyUI.
# ---------------------------------------------------------------------------
class SP_AnyPipe_Base:
    # The number of input/output ports. This will be overridden by child classes.
    NUM_PORTS = 0
    CATEGORY = 'SP-Nodes/Pipes'
    FUNCTION = "execute"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Dynamically generates the input definitions based on NUM_PORTS.
        """
        inputs = {
            "optional": {
                # The main pipe input, which can be modified.
                "cpipe_in": ("sp_anypipe",),
            }
        }
        # Dynamically create optional inputs: any_1_in, any_2_in, ...
        for i in range(1, cls.NUM_PORTS + 1):
            inputs["optional"][f"any_{i}_in"] = (AnyType('*'),)
        return inputs

    @classmethod
    def _get_return_definitions(cls):
        """
        A helper method to generate the return type and name tuples.
        This avoids code duplication in RETURN_TYPES and RETURN_NAMES.
        """
        return_types = ["sp_anypipe"]
        return_names = ["cpipe_out"]
        # Dynamically create outputs: any_1_out, any_2_out, ...
        for i in range(1, cls.NUM_PORTS + 1):
            return_types.append(AnyType('*'))
            return_names.append(f"any_{i}_out")
        return tuple(return_types), tuple(return_names)

    # Use properties to define RETURN_TYPES and RETURN_NAMES by calling the helper method.
    @classmethod
    @property
    def RETURN_TYPES(cls):
        return cls._get_return_definitions()[0]

    @classmethod
    @property
    def RETURN_NAMES(cls):
        return cls._get_return_definitions()[1]

    def execute(self, cpipe_in=None, **kwargs):
        """
        The main logic of the node. It merges the incoming pipe with individual inputs.
        """
        # 1. Initialize original values from the incoming pipe.
        # If no pipe is provided, all values default to None.
        originals = [None] * self.NUM_PORTS
        if cpipe_in is not None:
            # Safely copy values from the pipe, truncating or padding as needed.
            pipe_list = list(cpipe_in)[:self.NUM_PORTS]
            originals[:len(pipe_list)] = pipe_list

        # 2. Collect individual 'any_N_in' values from keyword arguments.
        inputs = [kwargs.get(f"any_{i}_in", None) for i in range(1, self.NUM_PORTS + 1)]

        # 3. Create the final list of values for the output pipe.
        # Priority is given to new individual inputs over the original pipe values.
        final_values = [
            inputs[i] if inputs[i] is not None else originals[i]
            for i in range(self.NUM_PORTS)
        ]

        # 4. Return the result.
        # The first item is the new pipe (as a list), followed by all its unpacked elements.
        return (final_values, *final_values)


# ---------------------------------------------------------------------------
# CONCRETE CLASSES: Inherit from the base and define the number of ports.
# These are the actual nodes that will appear in ComfyUI.
# ---------------------------------------------------------------------------
class SP_AnyPipe5(SP_AnyPipe_Base):
    NUM_PORTS = 5

class SP_AnyPipe10(SP_AnyPipe_Base):
    NUM_PORTS = 10

class SP_AnyPipe15(SP_AnyPipe_Base):
    NUM_PORTS = 15

class SP_AnyPipe20(SP_AnyPipe_Base):
    NUM_PORTS = 20

class SP_AnyPipe30(SP_AnyPipe_Base):
    NUM_PORTS = 30

class SP_AnyPipe40(SP_AnyPipe_Base):
    NUM_PORTS = 40

class SP_AnyPipe50(SP_AnyPipe_Base):
    NUM_PORTS = 50


# ---------------------------------------------------------------------------
# NODE REGISTRATION: This is how ComfyUI discovers the custom nodes.
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SP_AnyPipe5": SP_AnyPipe5,
    "SP_AnyPipe10": SP_AnyPipe10,
    "SP_AnyPipe15": SP_AnyPipe15,
    "SP_AnyPipe20": SP_AnyPipe20,
    "SP_AnyPipe30": SP_AnyPipe30,
    "SP_AnyPipe40": SP_AnyPipe40,
    "SP_AnyPipe50": SP_AnyPipe50,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SP_AnyPipe5": "SP Any Pipe [5 Ports]",
    "SP_AnyPipe10": "SP Any Pipe [10 Ports]",
    "SP_AnyPipe15": "SP Any Pipe [15 Ports]",
    "SP_AnyPipe20": "SP Any Pipe [20 Ports]",
    "SP_AnyPipe30": "SP Any Pipe [30 Ports]",
    "SP_AnyPipe40": "SP Any Pipe [40 Ports]",
    "SP_AnyPipe50": "SP Any Pipe [50 Ports]",
}