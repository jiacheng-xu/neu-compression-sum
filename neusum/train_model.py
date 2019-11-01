import time

from neusum.service.common_imports import *
from neusum.service.basic_service import *
from neusum.service.argparser import parse_arg_interface
from neusum.data.build_data import data_builder
from neusum.models.build_seq2idx import build_model
from neusum.training.build_trainer_seq2idx import build_trainer
import sys
import os
from neusum.service.shared_asset import *

if __name__ == '__main__':
    args = parse_arg_interface()
    if args.dbg:
        args.emb_dim = 11
        args.hid_dim = 37
        args.batch_size = 13
    if args.compression is False:
        args.validation_metric = "+sent_A"
    else:
        args.validation_metric = "+cp_A"
    fname = prepare_file_name(
        data=args.data_name,
        cmpres=args.compression,
        step=args.fix_edu_num,
        leadn=args.compress_leadn,
        lr=args.lr,
        dbg=args.dbg,
        alpha=args.alpha,

        optim=args.optim,
        schedule=args.schedule,
        avdTri=args.dec_avd_trigram_rep,
        trimSentOra=args.trim_sent_oracle,

        mult_orac_sampling=args.mult_orac_sampling,

        maxStep=args.max_dec_step,
        minStep=args.min_dec_step,

        train_fname=args.train_fname,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        # single_oracle=args.single_oracle,
        elmo=args.elmo,
        advCmp=args.aggressive_compression
    )
    args.fname = fname
    if args.data_name in ["cnn", "cnndm", "dm"]:
        args.vocab_save_to = os.path.join(args.abs_dir_root, 'cnndm_vocab')
    elif args.data_name == "nyt":
        args.vocab_save_to = os.path.join(args.abs_dir_root, 'nyt_vocab')
    if args.dbg:
        _logging_level = logging.DEBUG
        args.pretrain_embedding = None
    else:
        _logging_level = logging.INFO
    logger = prepare_global_logger(stdout_file_name=os.path.join(args.abs_dir_log, fname + '.log')
                                   , level=_logging_level)
    logger.info("AllenNLP version: {}".format(allennlp.__version__))
    logger.info("Parameters: {}".format(args))
    logger.info("Time: {}".format(time.ctime()))
    device = torch.device("cuda:{}".format(args.device_cuda_id) if not args.device_use_cpu else "cpu")
    set_device(device)

    args.device_cuda_id = -1 if args.device_use_cpu else args.device_cuda_id
    set_device_id(args.device_cuda_id)

    torch.cuda.set_device(args.device_cuda_id)

    train_data, dev_data, test_data, vocab, word_token_indexers = data_builder(dbg=args.dbg,
                                                                               lazy=args.lazy,
                                                                               dataset_dir=args.abs_dir_data,
                                                                               train_file=args.train_fname,
                                                                               dev_file=args.dev_fname,
                                                                               test_file=args.test_fname,
                                                                               single_oracle=args.single_oracle,
                                                                               fix_edu_num=args.fix_edu_num,
                                                                               trim_sent_oracle=args.trim_sent_oracle,
                                                                               save_to=args.vocab_save_to
                                                                               )

    if args.abs_dir_exp is not None and os.path.exists(os.path.join(args.abs_dir_exp, fname)) is False:
        os.mkdir(os.path.join(args.abs_dir_exp, fname))
        logger.info("Creating dir: {}".format(os.path.join(args.abs_dir_exp, fname)))

    model = build_model(
        vocab=vocab,
        embed_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        min_dec_step=args.min_dec_step,
        max_decoding_steps=args.max_dec_step,
        fix_edu_num=args.fix_edu_num,
        use_elmo=args.elmo,
        dropout=args.dropout,
        dropout_emb=args.dropout_emb,
        span_encoder_type=args.span_encoder_type,
        attn_type=args.attn_type,
        schedule_ratio_from_ground_truth=args.schedule,
        pretrain_embedding=args.pretrain_embedding,
        nenc_lay=args.nenc_lay,
        mult_orac_sampling=args.mult_orac_sampling,
        compression=args.compression,
        word_token_indexers=word_token_indexers,
        alpha=args.alpha,
        dbg=args.dbg,
        dec_avd_trigram_rep=args.dec_avd_trigram_rep,
        aggressive_compression=args.aggressive_compression,
        weight_alpha=args.weight_alpha,
        bias_alpha=args.bias_alpha,
        abs_board_file=args.abs_board_file,
        keep_threshold=args.keep_threshold,
        compress_leadn=args.compress_leadn,
        gather=args.gather,
        abs_dir_root=args.abs_dir_root,
        serilization_name="{}{}{}{}{}{}".format(args.data_name, args.compression,
                                              args.alpha, args.compress_leadn, args.elmo, args.max_dec_step),
        load_save_model=args.load_saved_model
    )

    trainer = build_trainer(train_data=train_data,
                            dev_data=dev_data,
                            test_data=test_data,
                            vocab=vocab,
                            device_id=args.device_cuda_id,
                            model=model,
                            optim_option=args.optim,
                            serialization_dir=os.path.join(args.abs_dir_exp, fname),
                            batch_size=args.batch_size,
                            eval_batch_size=args.eval_batch_size,
                            lr=args.lr,
                            patience=args.patience, nepo=args.epochs, grad_clipping=args.clip,
                            validation_metric=args.validation_metric,
                            validation_interval=args.validation_interval)
    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    logger.info("Metrics: {}".format(metrics))
    logger.info("Parameters: {}".format(args))
    print(metrics)
    print(fname)
    print(args)
    log_board_file(args.abs_board_file, args, metrics)
