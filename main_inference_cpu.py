import os
import time
import logging
import numpy as np
import torch
import tensorflow as tf
import cv2
from itertools import groupby
from typing import List, Tuple

from lib.config import get_cfg
from lib.engine.defaults import default_argument_parser, default_setup
from lib.model import SignModel
from lib.utils.ksl_utils import clean_ksl
from lib.transforms.transform_gen import CenterCrop, RandomCrop, Resize, ToTensorGen, TransformGen
from fvcore.transforms.transform import Transform, TransformList


def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = default_setup(cfg, args)

    return cfg

def arr2sen(sequences, vocab):
    index_to_word = {index: word for word, index in vocab.items()}
    return [[index_to_word.get(idx, '<unk>') for idx in sequence] for sequence in sequences]

def main(args):
    start = time.time()
    logger = logging.getLogger()

    cfg = setup(args)
    cfg.freeze()

    vocab = {'<si>': 0, '<unk>': 1, '<pad>': 2, '차내리다': 3, '곳': 4, '버스': 5, '내리다': 6, '맞다': 7, '전': 8, '지름길': 9,
             '송파': 10, '지하철': 11, '무엇': 12, '가다': 13, '방법': 14, '여기': 15, '목적': 16, '건너다': 17, '명동': 18, '보다': 19,
             '시청': 20, '신호등': 21, '저기': 22, '다음': 23, '도착': 24, '우회전': 25, '좌회전': 26, '찾다': 27, '길': 28}

    test1 = time.time()

    video_path = cfg.DATASET.VIDEO_ROOT
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    frames = np.array(frames)
    test_dataset, _ = transform_image(cfg, frames, is_train=False)

    end_data = time.time()
    print('data', end_data - start)

    model = SignModel(vocab)
    # model = model.cuda()

    end_model_set = time.time()
    print('model setting', end_model_set - end_data)
    assert os.path.isfile(cfg.RESUME), "Error: no checkpoint directory found!"
    checkpoint = torch.load(cfg.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    logger.info(
        "Loaded checkpoint from {}.  "
        "start_epoch: {cp[epoch]}  "
        "recoded WER: {cp[wer]:.3f} (best: {cp[best_wer]:.3f})".format(
            cfg.RESUME, cp=checkpoint
        )
    )

    device = torch.device("cpu")
    model = model.to(device)

    end_model_load = time.time()
    print('model load', end_model_load - end_model_set)

    end_test_dataset = time.time()
    print('test_dataset get', end_test_dataset - end_model_set)

    validate(model, test_dataset, vocab, device)
    end = time.time()

    print('Inference time: ', end - end_test_dataset, 's')
    print('Total time: ', end - start, 's')


def validate(model, val_dataset, vocab, device):
    logger = logging.getLogger()

    model.eval()

    all_glosses = []

    with torch.no_grad():
        val_dataset = val_dataset.unsqueeze(0).to(device).detach()
        print(val_dataset.shape, len(val_dataset))
        video_lengths = np.array([val_dataset.shape[1]])

        gloss_scores = model(val_dataset)  # (B, T, C)

        gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
        gloss_probs = gloss_probs.detach().numpy()  # (T, B, C)
        gloss_probs_tf = np.concatenate((gloss_probs[:, :, 1:], gloss_probs[:, :, 0, None]), axis=-1)
        sequence_length = video_lengths // 4

        ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
            inputs=gloss_probs_tf,
            sequence_length=sequence_length,
            beam_width=1,
            top_paths=1,
        )
        ctc_decode = ctc_decode[0]

        tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]  # (B, )
        for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
            tmp_gloss_sequences[dense_idx[0]].append(ctc_decode.values[value_idx].numpy() + 1)

        decoded_gloss_sequences = []
        for seq_idx in range(0, len(tmp_gloss_sequences)):
            decoded_gloss_sequences.append([x[0] for x in groupby(tmp_gloss_sequences[seq_idx])])
        all_glosses.extend(decoded_gloss_sequences)

    assert len(all_glosses) == len(val_dataset)
    decoded_gls = arr2sen(all_glosses, vocab)

    gls_hyp = [clean_ksl(" ".join(t)) for t in decoded_gls]
    print('Predicted gloss: ', gls_hyp)

def transform_image(cfg, img: np.ndarray, is_train: bool = True) -> Tuple[np.ndarray, TransformList]:
    """
    Apply a series of transformations to an image based on the given configuration.

    Args:
        cfg: Configuration object containing transformation parameters.
        img (ndarray): Input image as a numpy array.
        is_train (bool): Flag to indicate if the transformation is for training or validation.

    Returns:
        ndarray: The transformed image.
        TransformList: A list of transforms that were applied to the image.
    """
    logger = logging.getLogger(__name__)

    resize = cfg.DATASET.TRANSFORM.RESIZE_IMG
    ts = cfg.DATASET.TRANSFORM.TEMPORAL_SCALING
    crop = cfg.DATASET.TRANSFORM.CROP_SIZE
    tc = cfg.DATASET.TRANSFORM.TEMPORAL_CROP_RATIO
    norm_params = dict(mean=cfg.DATASET.TRANSFORM.MEAN, std=cfg.DATASET.TRANSFORM.STD)

    tfm_gens = [Resize(resize, temporal_scaling=ts, interp="trilinear")]
    tfm_gens.append(CenterCrop(crop, temporal_crop_ratio=tc))
    tfm_gens.append(ToTensorGen(normalizer=norm_params))

    for g in tfm_gens:
        assert isinstance(g, TransformGen), f"Expected TransformGen, got {type(g)}"

    check_dtype(img)

    tfms = []
    for g in tfm_gens:
        tfm = g.get_transform(img)
        assert isinstance(tfm, Transform), f"TransformGen {g} must return an instance of Transform! Got {tfm} instead"
        img = tfm.apply_image(img)
        tfms.append(tfm)

    return img, TransformList(tfms)


def check_dtype(img):
    assert isinstance(img, np.ndarray), "[TransformGen] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
            img.dtype == np.uint8
    ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [3, 4], img.ndim


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = 'configs/config_inference.yaml'

    main(args)
