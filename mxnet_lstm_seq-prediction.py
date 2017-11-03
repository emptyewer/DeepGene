import os
import numpy as np
import mxnet as mx
import argparse
import logging

parser = argparse.ArgumentParser(description="Train RNN on Coding Sequence",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default=False, action='store_true',
                    help='whether to do testing instead of training')
parser.add_argument('--data-prefix', type=str, default='data',
                    help='Prefix for input train and val data')
parser.add_argument('--model-prefix', type=str, default=None,
                    help='path to save/load model')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='load from epoch')
parser.add_argument('--num-lstm-layers', type=int, default=2,
                    help='number of stacked LSTM RNN layers')
parser.add_argument('--num-lstm-units', type=int, default=1500,
                    help='number of LSTM units')
parser.add_argument('--num-embed', type=int, default=300,
                    help='embedding layer size. Best dimension is 300!')
parser.add_argument('--bidirectional', type=bool, default=False,
                    help='whether to use bidirectional layers')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
parser.add_argument('--seed', type=int, default=1982,
                    help='random seed to use. Default=1982.')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='Save every n epoch Default=1.')    
parser.add_argument('--bucket-size', type=int, default=35,
                    help='Size of the buckets Default=35')                    
# When training a deep, complex model *on multiple GPUs* it's recommended to
# stack fused RNN cells (one layer per cell) together instead of one with all
# layers. The reason is that fused RNN cells don't set gradients to be ready
# until the computation for the entire layer is completed. Breaking a
# multi-layer fused RNN cell into several one-layer ones allows gradients to be
# processed ealier. This reduces communication overhead, especially with
# multiple GPUs.
parser.add_argument('--stack-rnn', default=False,
                    help='stack fused RNN cells to reduce communication overhead')
parser.add_argument('--dropout', type=float, default='0.9',
                    help='dropout probability (1.0 - keep probability)')

# buckets = [10, 20, 30, 40, 50, 60]

start_label = 1
invalid_label = 0

def tokenize_text(fname, vocab=None, invalid_label=-1, invalid_key='XXX', start_label=0):
    lines = open(fname).readlines()
    lines = [filter(None, i.strip().split(' ')) for i in lines]
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, 
                                               invalid_key=invalid_key, start_label=start_label)
    return sentences, vocab

def get_data(layout):
    train_sent, vocab = tokenize_text("./data/%s_train.txt" % args.data_prefix, start_label=start_label,
                                      invalid_label=invalid_label)
    val_sent, _ = tokenize_text("./data/%s_test.txt" % args.data_prefix, vocab=vocab, start_label=start_label,
                                invalid_label=invalid_label)

    data_train  = mx.rnn.BucketSentenceIter(train_sent, args.batch_size, buckets=[args.bucket_size],
                                            invalid_label=invalid_label, layout=layout)
    data_val    = mx.rnn.BucketSentenceIter(val_sent, args.batch_size, buckets=[args.bucket_size],
                                            invalid_label=invalid_label, layout=layout)
    return data_train, data_val, vocab


def train(args):
    data_train, data_val, vocab = get_data('TN')
    logging.info("Vocab Size %d" % len(vocab))
    logging.info("Vocabulary: \n {}".format(vocab))
    params_name = 'params/%s/%s-%s-%d_nlu-%d_nll-%d_nem-%d_bidir-%s_opt-%.3f_lr-%.1f_mom-%.5f_wd-%d_seed-bucket%d' % (args.data_prefix, args.model_prefix, args.data_prefix, args.num_lstm_units, 
                                                                                                                      args.num_lstm_layers, args.num_embed, args.bidirectional, 
                                                                                                                      args.optimizer, args.lr, args.mom, args.wd, args.seed, 
                                                                                                                      args.bucket_size)
    if not os.path.exists("params/%s" % args.data_prefix):
        os.makedirs("params/%s" % args.data_prefix)

    logging.info("Writing params to file %s", params_name)
    if args.stack_rnn:
        cell = mx.rnn.SequentialRNNCell()
        for i in range(args.num_lstm_layers):
            cell.add(mx.rnn.FusedRNNCell(args.num_lstm_units, num_layers=1,
                                         mode='lstm', prefix='lstm_l%d'%i,
                                         bidirectional=args.bidirectional))
            if args.dropout > 0 and i < args.num_lstm_layers - 1:
                cell.add(mx.rnn.DropoutCell(args.dropout, prefix='lstm_d%d'%i))
    else:
        cell = mx.rnn.FusedRNNCell(args.num_lstm_units, num_layers=args.num_lstm_layers, dropout=args.dropout,
                                   mode='lstm', bidirectional=args.bidirectional)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab), output_dim=args.num_embed,name='embed')
        output, _ = cell.unroll(seq_len, inputs=embed, merge_outputs=True, layout='TNC')
        pred = mx.sym.Reshape(output, shape=(-1, args.num_lstm_units*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')
        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=data_train.default_bucket_key,
                                   context=contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(cell, params_name, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = ['ce', mx.metric.Perplexity(invalid_label)],
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = opt_params,
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = args.load_epoch,
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches, auto_reset=False),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, params_name, args.save_epoch)
                              if args.model_prefix else None)

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    logging.info(args)

    if args.num_lstm_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    mx.random.seed(args.seed)
    train(args)
    
        
