import prompting_classes.zero_shot_prompt_convertor,prompting_classes.cot_prompt_convertor
from config import config
from prompting_classes.common import converter_registry


def create_prompt_from_config(config):
    # Get the class reference from the registry
    PromptClass = converter_registry.get(config.prompt_creator)

    if not PromptClass:
        raise ValueError(f"Invalid class name: {config.prompt_creator}")

    return PromptClass

prompt_cls = create_prompt_from_config(config)
