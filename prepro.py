#
# A wrapper script that trains the SELDnet and SELD-TCN.
# The training stops when the SELD error (check paper) stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_data_generator
import evaluation_metrics
import keras_model
import parameter
import utils
import time
from IPython import embed
plot.switch_backend('agg')
np.set_printoptions(threshold=sys.maxsize)


def collect_test_labels(_data_gen_test, _data_out, classification_mode, quick_test):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[1]
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa


def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _epoch_metric_loss):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(311)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(nb_epoch), _epoch_metric_loss, label='metric')
    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: job_id - (optional) all the output files will be uniquely represented with this. (default) 1
        second input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) uses default parameters
    """
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two inputs')
        print('\t>> python seld.py <job-id> <task-id>')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')
    # use parameter set defined by user
    task_id = '1' if len(argv) < 3 else argv[-1]
    params = parameter.get_params(task_id)

    job_id = 1 if len(argv) < 2 else argv[1]

    model_dir = 'models/'
    utils.create_folder(model_dir)
    unique_name = '{}_ov{}_split{}_{}{}_3d{}_{}'.format(
        params['dataset'], params['overlap'], params['split'], params['mode'], params['weakness'],
        int(params['cnn_3d']), job_id
    )
    unique_name = os.path.join(model_dir, unique_name)
    print("unique_name: {}\n".format(unique_name))

    data_gen_train = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='train', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only']
    )

    data_gen_test = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='test', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only'], shuffle=False
    )

    data_in, data_out = data_gen_train.get_data_sizes()
    print(
        'FEATURES:\n'
        '\tdata_in: {}\n'
        '\tdata_out: {}\n'.format(
            data_in, data_out
        )
    )

    gt = collect_test_labels(data_gen_test, data_out, params['mode'], params['quick_test'])
    sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
    doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
