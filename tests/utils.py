from nazuna.task_runners import EvalTaskRunner
import toml


def set_training(model, training: bool):
    if training:
        model.train()
    else:
        model.eval()


def create_from_doc(model_cls, device):
    doc = model_cls.__doc__.splitlines()
    key = f'[definitions.{model_cls.__name__}]'
    flag = False
    indent = None
    text = ''
    for line in doc:
        if flag:
            if line.strip() == '```':
                break
            text += line[indent:] + '\n'
        if key in line:
            indent = line.find(key)
            text += line[indent:] + '\n'
            flag = True
    print(text)
    d = toml.loads(text)['definitions'][model_cls.__name__]
    cls_, params_ = EvalTaskRunner.extract_model_config(d)
    assert cls_ is model_cls
    model = cls_(device, **params_)
    assert isinstance(model, model_cls)
    return model
