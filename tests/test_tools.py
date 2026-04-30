from nazuna.tools import parse_param_conf
from nazuna.models.patchtst import PatchTST


def test_parse_param_conf():
    text = parse_param_conf(PatchTST, {
        'seq_len': 96, 'pred_len': 24, 'c_in': 11,
    }, n_derived=2)
    print(text)
    assert text.startswith('[definitions.PatchTST]')
