# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from signjoey.search import beam_search, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    SIL_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import (
    freeze_params,
    subsequent_mask,
    wait_K_mask_generate,
    memory_mask_generator,
    gen_reencoder_mask_from_wait_k_mask,
    gen_wait_k_mask_from_boundary_possibility,
)
from signjoey.loss import XentLoss
from torch import Tensor
from typing import Union
from torchtext.vocab import Vectors


class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        gloss_output_layer: nn.Module,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        fire_hidden: nn.Linear,
        fire_layer: nn.Linear,
        nar_decoder: TransformerEncoder,
        nar_decoder_fc: nn.Linear,
        wait_k_method: str,
        do_recognition: bool = True,
        do_translation: bool = True,
        wait_k: int = 1,
        static_gls_len: int = 0,
        is_reencoder: bool = True,
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()

        self.wait_k = wait_k
        self.static_gls_len = static_gls_len
        self.is_reencoder = is_reencoder

        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation
        self.wait_k_method = wait_k_method

        self.fire_hidden = fire_hidden
        self.fire_layer = fire_layer
        self.nar_decoder = nar_decoder
        self.nar_decoder_fc = nar_decoder_fc
        self.nar_loss_func = XentLoss(pad_index=self.gls_vocab.stoi[PAD_TOKEN])

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        gls_lengths: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """
        predict_num = None
        nar_output = None
        boundary_possibility = None
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn,
            sgn_mask=sgn_mask,
            sgn_length=sgn_lengths,
            encoder=self.encoder,
            sgn_embed=self.sgn_embed,
            encoder_mask=subsequent_mask(sgn_mask.shape[-1]).type_as(sgn_mask),
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            if self.static_gls_len <= 0:
                if self.wait_k_method == "fire":
                    boundary_hidden = (
                        torch.relu(self.fire_hidden(encoder_output)) + encoder_output
                    )
                    boundary_output = self.fire_layer(boundary_hidden)
                    # shape: batch_size,max_sign
                    boundary_possibility = torch.sigmoid(boundary_output).squeeze()
                    predict_num = torch.sum(boundary_possibility, dim=-1)
                    weight_ratio = (gls_lengths / predict_num).unsqueeze(-1)
                    boundary_possibility = boundary_possibility * weight_ratio
                    (
                        wait_k_mask,
                        weighted_encoder_output,
                        nar_mask,
                    ) = memory_mask_generator(
                        gls_length=int(gls_lengths.max().item()),
                        txt_length=unroll_steps,
                        device=txt_input.device,
                        encoder_output=encoder_output,
                        wait_k=self.wait_k,
                        weight_set=boundary_possibility,
                    )
                    # 辅助任务
                    nar_output, _ = self.nar_decoder(
                        embed_src=weighted_encoder_output,
                        src_length=None,
                        mask=nar_mask.unsqueeze(1),
                    )
                    nar_output = self.nar_decoder_fc(nar_output)
                else:
                    raise ValueError("unknown wait k method")
            else:
                wait_k_mask = wait_K_mask_generate(
                    target_length=unroll_steps,
                    static_gls_len=self.static_gls_len,
                    batch_size=encoder_output.shape[0],
                    source_length=encoder_output.shape[1],
                    device=txt_input.device,
                    K=self.wait_k,
                )
            # generate reencoder mask form waitk mask
            if self.is_reencoder:
                reencoder_mask = gen_reencoder_mask_from_wait_k_mask(wait_k_mask)
                re_encoder_output, encoder_hidden = self.encode(
                    sgn=sgn,
                    sgn_mask=sgn_mask,
                    sgn_length=sgn_lengths,
                    encoder=self.encoder,
                    sgn_embed=self.sgn_embed,
                    encoder_mask=reencoder_mask,
                )
                encoder_output = re_encoder_output + encoder_output

            wait_k_sgn_mask = sgn_mask & wait_k_mask
            # print(sgn_mask[-1,0,:])
            # print(wait_k_sgn_mask[-1, 0, :])
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=wait_k_sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None
            predict_num = None
            nar_output = None

        return (
            decoder_outputs,
            gloss_probabilities,
            predict_num,
            nar_output,
            boundary_possibility,
        )

    def encode(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_length: Tensor,
        encoder_mask: Tensor,
        encoder: Encoder,
        sgn_embed: SpatialEmbeddings,
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        # 同声传译避免看到后面的内容
        att_mask = sgn_mask & encoder_mask

        return encoder(
            embed_src=sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=att_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def get_loss_for_batch(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable

        # Do a forward pass
        (
            decoder_outputs,
            gloss_probabilities,
            predict_num,
            nar_output,
            boundary_possibility,
        ) = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
            gls_lengths=batch.gls_lengths,
        )

        if self.do_recognition:
            assert gloss_probabilities is not None
            # Calculate Recognition Loss
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        if self.do_translation:
            nar_loss = 0
            boundary_loss = 0
            assert decoder_outputs is not None
            word_outputs, _, _, _ = decoder_outputs
            # Calculate Translation Loss
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch.txt)
                * translation_loss_weight
            )
            if self.wait_k_method == "fire":
                len_loss = torch.mean((predict_num - batch.gls_lengths) ** 2)
                nar_gloss_probs = F.log_softmax(nar_output, dim=-1)
                nar_loss = self.nar_loss_func(nar_gloss_probs, batch.gls)
                nar_loss = nar_loss / batch.num_seqs
                nar_loss = nar_loss + len_loss
        else:
            translation_loss = None
            nar_loss = None

        return recognition_loss, translation_loss, nar_loss, boundary_loss

    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        wait_k_mask = None
        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_length=batch.sgn_lengths,
            encoder=self.encoder,
            sgn_embed=self.sgn_embed,
            encoder_mask=subsequent_mask(batch.sgn_mask.shape[-1]).type_as(
                batch.sgn_mask
            ),
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            if self.static_gls_len <= 0:
                if self.wait_k_method == "fire":
                    boundary_hidden = (
                        torch.relu(self.fire_hidden(encoder_output)) + encoder_output
                    )
                    boundary_output = self.fire_layer(boundary_hidden)
                    boundary_possibility = torch.sigmoid(boundary_output).squeeze()
                    wait_k_mask, _, _ = memory_mask_generator(
                        txt_length=translation_max_output_length,
                        wait_k=self.wait_k,
                        device=encoder_output.device,
                        weight_set=boundary_possibility,
                    )
                else:
                    raise ValueError("unknown wait k method")
            else:
                wait_k_mask = wait_K_mask_generate(
                    target_length=translation_max_output_length,
                    static_gls_len=self.static_gls_len,
                    batch_size=encoder_output.shape[0],
                    source_length=encoder_output.shape[1],
                    device=encoder_output.device,
                    K=self.wait_k,
                )
            # generate reencoder mask form waitk mask
            if self.is_reencoder:
                reencoder_mask = gen_reencoder_mask_from_wait_k_mask(wait_k_mask)
                re_encoder_output, encoder_hidden = self.encode(
                    sgn=batch.sgn,
                    sgn_mask=batch.sgn_mask,
                    sgn_length=batch.sgn_lengths,
                    encoder=self.encoder,
                    sgn_embed=self.sgn_embed,
                    encoder_mask=reencoder_mask,
                )
                encoder_output = re_encoder_output + encoder_output

            wait_k_sgn_mask = batch.sgn_mask & wait_k_mask
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=wait_k_sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=wait_k_sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return (
            decoded_gloss_sequences,
            stacked_txt_output,
            stacked_attention_scores,
            wait_k_mask,
        )

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param gls_vocab: sign gloss vocabulary
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    """
    wait_k = cfg.get("wait_k", 1)
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    if do_recognition:
        gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))
        if cfg["encoder"].get("freeze", False):
            freeze_params(gloss_output_layer)
    else:
        gloss_output_layer = None

    # build decoder and word embeddings
    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
        if cfg["decoder"].get("type", "recurrent") == "transformer":
            decoder = TransformerDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
        else:
            decoder = RecurrentDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
    else:
        txt_embed = None
        decoder = None

    if cfg.get("wait_k_method", "") == "fire":
        boundary_hidden = nn.Linear(
            cfg["encoder"]["hidden_size"], cfg["encoder"]["hidden_size"]
        )
        boundary_layer = nn.Linear(cfg["encoder"]["hidden_size"], 1)
        nar_decoder = TransformerEncoder(
            **cfg["fire_decoder"],
            emb_dropout=cfg["fire_decoder"].get("embeddings_dropout", 0.1),
        )
        nar_decoder_fc = nn.Linear(cfg["fire_decoder"]["hidden_size"], len(gls_vocab))
    else:
        boundary_hidden = None
        boundary_layer = None
        nar_decoder = None
        nar_decoder_fc = None

    model: SignModel = SignModel(
        encoder=encoder,
        gloss_output_layer=gloss_output_layer,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        fire_hidden=boundary_hidden,
        fire_layer=boundary_layer,
        nar_decoder=nar_decoder,
        nar_decoder_fc=nar_decoder_fc,
        wait_k_method=cfg["wait_k_method"],
        do_recognition=do_recognition,
        do_translation=do_translation,
        wait_k=wait_k,
        static_gls_len=cfg.get("static_gls_len", 0),
        is_reencoder=cfg.get("is_reencoder", True),
    )

    if do_translation:
        # tie softmax layer with txt embeddings
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    if do_translation:
        vectors_path = cfg.get("word_vectors", "")
        if vectors_path:
            vectors_path = Path(vectors_path).expanduser()
            pretrain_vector = Vectors(
                name=vectors_path.name,
                cache=str(vectors_path.parent),
            )
            for i, token in enumerate(txt_vocab.itos):
                if not (pretrain_vector[token.strip()] == 0).all():
                    model.txt_embed.lut.weight.data[i][
                        : pretrain_vector.dim
                    ] = pretrain_vector[token.strip()]

    return model
