import math
from dataclasses import dataclass
from typing import Any

import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import numpy as np

from carp.configs import DataPipelineConfig
from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder


def extract_augmenter_args(config: dict, key: str) -> Tuple[bool, dict[str, Any] or None]:
    """
    Returns arguments for augmenter specified by key.

    Notes:
    An empty dict in .yml is valid, cause augmenters can take no args (i.e. default kwargs).
    If it doesn't exist, it should NOT be added, and should be set to None.

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
            else:  # When the config file explicitly specifies not to use the augment via a bool
                return False, None
        return True, arg
    else:
        return False, None


@dataclass
class NLPAugConfig:
    augmenter_flow: naf.Pipeline
    augment_passages: bool
    augment_reviews: bool
    augmentation_likelihood: float


def get_nlpaug_config(path: str) -> NLPAugConfig:
    pipeline_config: DataPipelineConfig = DataPipelineConfig.load_yaml(path)
    args: dict = pipeline_config.args
    augment_reviews: bool = args.get('augment_reviews', False)
    augment_passages: bool = args.get('augment_passages', False)
    augmentation_likelihood: float = args.get('augmentation_likelihood', None)
    if augmentation_likelihood > 1.0 or augmentation_likelihood < 0.0:
        raise ValueError(f'Augmentation likelihood should be between 0 and 1, received: {augmentation_likelihood}')

    if augment_passages and augment_reviews or (not augment_passages and not augment_reviews):
        print('WARNING: Both passages and reviews are being augmented.')

    # Sentence Augmentations

    has_cwe, context_embedding_args = extract_augmenter_args(args, 'augment_context_embedding')
    has_lambada, lambada_args = extract_augmenter_args(args, 'augment_lambada')
    model_directory: str = args.get('lambada_model_directory', None)
    has_abs_sum, abs_sum_args = extract_augmenter_args(args, 'augment_abstractive_summarization')
    has_rand, random_aug_args = extract_augmenter_args(args, 'augment_random_behavior')

    # Word Augmentations
    has_spelling, spelling_args = extract_augmenter_args(args, 'augment_spelling')
    has_split, split_args = extract_augmenter_args(args, 'augment_split')
    has_synonym, synonym_args = extract_augmenter_args(args, 'augment_synonym')
    has_tfidf, tfidf_args = extract_augmenter_args(args, 'augment_tfidf')
    has_backtranslation, backtranslation_args = extract_augmenter_args(args, 'augment_backtranslation')

    # Sentence Augmentations
    contextual_word_embedding_aug = nas.ContextualWordEmbsForSentenceAug(**context_embedding_args) if has_cwe else None
    lambada_aug = nas.LambadaAug(model_directory, **lambada_args) if has_lambada else None
    abstractive_summarization_aug = nas.AbstSummAug(**abs_sum_args) if has_abs_sum else None
    random_sentence_augmenter = nas.RandomSentAug(**random_aug_args) if has_rand else None
    # Word Augmentations
    spelling_augmenter = naw.SpellingAug(**spelling_args) if has_spelling else None
    split_augmenter = naw.SpellingAug(**split_args) if has_split else None
    synonym_augmenter = naw.SynonymAug(**synonym_args) if has_synonym else None
    tfidf_augmenter = naw.TfIdfAug(**tfidf_args) if has_tfidf else None
    backtranslation_augmenter = naw.BackTranslationAug(**backtranslation_args) if has_backtranslation else None

    augmenters = [contextual_word_embedding_aug, lambada_aug, abstractive_summarization_aug,
                  random_sentence_augmenter, spelling_augmenter, split_augmenter, synonym_augmenter, tfidf_augmenter,
                  backtranslation_augmenter]

    # Uses each augmenter with a certain probability, default is uniform across all augmenters passed in
    augmenter_pipeline: naf.Pipeline = naf.Sometimes([aug for aug in augmenters if aug is not None])
    return NLPAugConfig(augmenter_pipeline, augment_passages, augment_reviews, augmentation_likelihood)


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
            augmenter = pipeline_config.augmenter_flow

            if pipeline_config.augment_reviews:
                num_aug_data = math.ceil(len(passages) * pipeline_config.augmentation_likelihood)
                indices_of_data_to_augment = np.random.choice(np.arange(len(augmented_reviews)), size=num_aug_data,
                                                              replace=False)
                data_to_augment = augmented_reviews[indices_of_data_to_augment]
                # Split into number of augmenters?
                augmented_slice = augmenter.augment(list(data_to_augment))
                augmented_reviews[indices_of_data_to_augment] = augmented_slice

            if pipeline_config.augment_passages:
                augmented_passages = augmenter.augment(list(augmented_passages))

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
