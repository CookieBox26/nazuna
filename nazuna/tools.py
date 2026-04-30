import re


def parse_param_conf(model_cls, overrides=None, n_derived=0):
    doc = model_cls.__doc__.splitlines()
    key = f'[definitions.{model_cls.__name__}]'
    flag = False
    indent = None
    text = ''
    for line in doc:
        if flag:
            if line.strip() == '```':
                break
            s = line[indent:]
            text += s + '\n'
        if key in line:
            indent = line.find(key)
            text += line[indent:] + '\n'
            flag = True

    if overrides is not None:
        for k, v_raw in overrides.items():
            pattern = rf'(?m)^({re.escape(k)}\s*=\s*)([^#\n]*)(\s*(?:#.*)?)$'
            v = f'"{v_raw}"' if type(v_raw) is str else v_raw
            text = re.sub(pattern, rf'\g<1>{v}  \g<3>', text)

    for i in range(n_derived):
        text += '\n'
        text += f'[definitions.{model_cls.__name__}_{i:02}]\n'
        text += f'base = "{model_cls.__name__}"\n'
        text += f'# [definitions.{model_cls.__name__}_{i:02}.params]\n'
    return text
