from dataclasses import dataclass
from typing import Any

import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from torchtyping import TensorType

from carp.configs import DataPipelineConfig
from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder


def extract_augmenter_args(config: dict, key: str) -> Tuple[bool, dict[str, Any] or None]:
    """

    Args:
        key: key of augmenter to extract arguments of
        config: a dictionary where each key represents an augmenter, and each value is either a list of arguments
        or a bool

    Returns:
        A tuple of :
            a bool indicating whether augmenter is in there to begin with

            a dictionary (potentially empty) which contains kwargs for the augmenter.
                None if the augmenter is not in config


    """
    if key in config:
        arg: dict or bool = config.get(key)
        if type(arg) is bool:
            if arg:
                arg = dict()
            else: # When the config file explicitly specifies not to use the augment via a bool
                return False, None
        return True, arg
    else:
        return False, None


@dataclass
class NLPAugConfig:
    augmenter_sequence: naf.Sequential
    augment_passages: bool
    augment_reviews: bool


def get_nlpaug_config(path: str) -> NLPAugConfig:
    pipeline_config: DataPipelineConfig = DataPipelineConfig.load_yaml(path)
    args: dict = pipeline_config.args
    augment_reviews: bool = args.get('augment_reviews', False)
    augment_passages: bool = args.get('augment_passages', False)

    if augment_passages and augment_reviews or (not augment_passages and not augment_reviews):
        print('WARNING: Both passages and reviews are being augmented.')

    # An empty dict in .yml is valid, cause this augmenter can take no args. If it doesn't exist, it should
    #  NOT be added, and should be set to None. This is why extract_augmenter_args is abstracted into its own fn
    has_cwe, augment_context_embedding = extract_augmenter_args(args, 'augment_context_embedding')
    has_lambada, augment_lambada = extract_augmenter_args(args, 'augment_lambada')
    model_directory: str = args.get('lambada_model_directory', None)
    has_abs_sum, augment_abstractive_summarization = extract_augmenter_args(args,
                                                                            'augment_abstractive_summarization')
    has_rand, augment_random_behavior = extract_augmenter_args(args, 'augment_random_behavior')

    # Initialize augmenter objects with extracted kwargs
    contextual_word_embedding_aug = nas.ContextualWordEmbsForSentenceAug(**augment_context_embedding) if has_cwe \
        else None
    # We don't check for both augment_lambada and model_directory so that it throws an error if a path is not
    # specified and doesn't fail silently
    lambada_aug = nas.LambadaAug(model_directory, **augment_lambada) if has_lambada else None
    abstractive_summarization_aug = nas.AbstSummAug(**augment_abstractive_summarization) if has_abs_sum else None
    random_sentence_augmenter = nas.RandomSentAug(**augment_random_behavior) if has_rand else None

    augmenters = [contextual_word_embedding_aug, lambada_aug, abstractive_summarization_aug,
                  random_sentence_augmenter]

    augmenter_pipeline: naf.Sequential = naf.Sequential([aug for aug in augmenters if aug is not None])
    return NLPAugConfig(augmenter_pipeline, augment_passages, augment_reviews)


def augment_labels_tokens() -> TensorType:
    # TODO
    pass


@register_datapipeline
class AugDataPipeline(BaseDataPipeline):
    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    def __init__(
            self,
            config: TrainConfig,
            path: str = "dataset"
    ):
        super(AugDataPipeline, self).__init__(config, path)

    @staticmethod
    def tokenizer_factory(_tok: Callable, _config: TrainConfig, encoder: BaseEncoder) -> Callable:
        """Function factory that creates a collate function for use with a torch.util.data.Dataloader

        Args:
            _tok:
            encoder:
            _config:

        Returns:
            Callable: A function that will take a batch of string tuples and tokenize them properly.
        """

        pipeline_config: NLPAugConfig = get_nlpaug_config(_config.pipeline_config_path)

        @typechecked
        def collate(
                data: Iterable[Tuple[str, str]]
        ) -> Tuple[BatchElement, BatchElement]:

            passages, reviews = zip(*data)
            # Augmentation happens here
            augmented_passages: list[str] = passages
            augmented_reviews: list[str] = reviews
            # TODO: Add Flow args to config to pass into augmenterseqeunce.augment
            # TODO: Should we be able to augment both?
            # TODO: Superclass kwargs shouild go to config file
            if pipeline_config.augment_reviews:
                augmented_reviews = pipeline_config.augmenter_sequence.augment(augmented_reviews)

            if pipeline_config.augment_passages:
                augmented_passages = pipeline_config.augmenter_sequence.augment(augmented_passages)
            # [a, b, c] --> [1, 2, 3]
            # [a+, b+, c+] --> [1, 2, 3]
            pass_tokens, rev_tokens = _tok(augmented_passages), _tok(augmented_reviews)
            pass_masks = pass_tokens["attention_mask"]
            rev_masks = rev_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]
            rev_tokens = rev_tokens["input_ids"]

            return (
                BatchElement(pass_tokens, pass_masks),
                BatchElement(rev_tokens, rev_masks),
            )

        return collate
