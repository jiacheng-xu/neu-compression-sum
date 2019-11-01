from neusum.models.seq2idx import Seq2IdxSum

from allennlp.nn import InitializerApplicator
from allennlp.nn.regularizers import L1Regularizer, L2Regularizer, RegularizerApplicator
from neusum.service.shared_asset import get_device

import torch

print(torch.__version__)


def build_model(
        vocab, embed_dim: int = 100,
        hid_dim: int = 100,
        min_dec_step: int = 2,
        max_decoding_steps: int = 3,
        fix_edu_num: int = -1,
        use_elmo: bool = False,
        dropout=0.5,
        dropout_emb=0.2, span_encoder_type='self_attentive',
        attn_type='dot',
        schedule_ratio_from_ground_truth=0.7,
        pretrain_embedding=None,
        nenc_lay: int = 1,
        mult_orac_sampling: bool = True,
        compression: bool = True,
        word_token_indexers=None,
        alpha: float = 1.0,
        dbg: bool = False,
        dec_avd_trigram_rep: bool = True,
        aggressive_compression: int = -1,
        keep_threshold: float = 0.5,
        weight_alpha=0.0,
        bias_alpha=0.0,
        abs_board_file: str = "/home/cc/exComp/board.txt",
        compress_leadn=-1,
        gather='mean',
        abs_dir_root: str = "/scratch/cluster/jcxu",
        serilization_name="",
        load_save_model: str = None
):
    model = Seq2IdxSum(
        vocab=vocab,
        word_embedding_dim=embed_dim,
        hidden_dim=hid_dim, min_dec_step=min_dec_step,
        max_decoding_steps=max_decoding_steps,
        fix_edu_num=fix_edu_num,
        use_elmo=use_elmo, span_encoder_type=span_encoder_type,
        dropout=dropout, dropout_emb=dropout_emb,
        attn_type=attn_type,
        schedule_ratio_from_ground_truth=schedule_ratio_from_ground_truth,
        pretrain_embedding_file=pretrain_embedding,
        nenc_lay=nenc_lay,
        mult_orac_sampling=mult_orac_sampling,
        word_token_indexers=word_token_indexers,
        compression=compression, alpha=alpha,
        dbg=dbg,
        dec_avd_trigram_rep=dec_avd_trigram_rep,
        aggressive_compression=aggressive_compression,
        keep_threshold=keep_threshold,
        regularizer=RegularizerApplicator([("weight", L2Regularizer(weight_alpha)),
                                           ("bias", L1Regularizer(bias_alpha))]),
        abs_board_file=abs_board_file,
        gather=gather,
        compress_leadn=compress_leadn,
        abs_dir_root=abs_dir_root,
        serilization_name=serilization_name
    )
    if load_save_model:
        model.load_state_dict(torch.load(load_save_model, map_location=get_device()))
    #         `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

    # model = torch.nn.DataParallel(model)
    device = get_device()
    model = model.to(device)
    return model
