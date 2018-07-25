import logging
import numpy as np

def log_with_title(arg, section_title):
    print("%s" % section_title)
    if isinstance(arg, dict):
        for k in sorted(arg):
            print("%s: %s" % (k, arg[k]))
    else:
        for item in arg:
            print(item)

def sanity_check(net_symbol, train_iter, batch_size):
    multi_card_to_single_card = lambda x: (batch_size, ) + x[1:]
    single_card_input_shape = {k: multi_card_to_single_card(v) for k, v in
                               dict(train_iter.provide_data + train_iter.provide_label).items()}
    _, out_shape, _ = net_symbol.get_internals().infer_shape(**single_card_input_shape)
    omemory = [np.prod(oshape) * 4 / float(1 << 20) for oshape in out_shape]
    out_names = net_symbol.get_internals().list_outputs()
    log_with_title(list(zip(out_names, out_shape, ["%.1fMB" % _ for _ in omemory])), "output shape: ")

    return single_card_input_shape
