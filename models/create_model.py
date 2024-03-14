from models.DiT.models import DiT_models
from models.taskrouting import TaskRouter
from models.MoE import TaskMoE


def create_model(model_config, routing_config):
    """
    Create various architectures from model_config
    """
    if model_config.name in DiT_models.keys():
        if routing_config.name == 'DTR':
            model = DiT_models[model_config.name](
                input_size=model_config.param.latent_size,
                num_classes=model_config.param.num_classes,
                router=TaskRouter,
                routing_config=routing_config
            )
        elif routing_config.name == 'DMoE':
            model = DiT_models[model_config.name](
                input_size=model_config.param.latent_size,
                num_classes=model_config.param.num_classes,
                router=TaskMoE,
                routing_config=routing_config
            )
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return model
