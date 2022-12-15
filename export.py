import os.path

import numpy as np
from mindspore import load_checkpoint, load_param_into_net, export, Tensor, context
from src.dinknet import DinkNet34, DinkNet50
from src.model_utils.config import config

if __name__ == "__main__":
    print(config)

    if config.enable_modelarts:
        import moxing as mox
        pretrained_ckpt_path = "/cache/origin_weights/pretrained_model.ckpt"
        mox.file.copy_parallel(config.pretrained_ckpt, pretrained_ckpt_path)
        trained_ckpt_path = "/cache/origin_weights/trained_model.ckpt"
        mox.file.copy_parallel(config.trained_ckpt, trained_ckpt_path)
        local_train_url = "/cache/export_out/"
        target = '../../../export_out/'
        mox.file.make_dirs(target)
        print('path[/cache/export_out] exist:', mox.file.exists(target))
    else:
        trained_ckpt_path = config.trained_ckpt
        local_train_url = './'

    BATCH_SIZE = config.batch_size

    # set context
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    # --------------------------------------
    if config.model_name == 'dinknet34':
        net = DinkNet34()
    else:
        net = DinkNet50()
    param_dict = load_checkpoint(trained_ckpt_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([BATCH_SIZE, config.num_channels, config.width, config.height], np.float32))
    export(net, input_arr, file_name=os.path.join(local_train_url, config.file_name), file_format=config.file_format)

    if config.enable_modelarts:
        mox.file.copy_parallel(local_train_url, config.train_url)
