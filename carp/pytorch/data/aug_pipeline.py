from dataclasses import dataclass
from typing import Any

from carp.configs import TrainConfig
from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder

import nlpaug.flow as naf
import nlpaug.augmenter.sentence as nas


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
    aug: dict or bool = config.get(key, None)
    if type(aug) is bool:
        return True, dict()

    return aug is None, aug


@dataclass
class NLPAugConfig:
    augmenter_sequence: naf.Sequential
    augment_passages: bool
    augment_reviews: bool


@register_datapipeline
class AugDataPipeline(BaseDataPipeline):
    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    nlpaug_config: NLPAugConfig = None

    def __init__(
            self,
            dupe_protection: bool = True,
            path: str = "dataset",
            aug_config: TrainConfig = None
    ):
        # TODO: I don't have the data locally (or the path, by extension), so I have to comment this out for it to run
        # super(AugDataPipeline, self).__init__(dupe_protection, path)
        # Probably not both should be true
        config: dict = aug_config.aug_args
        augment_reviews: bool = config.get('augment_reviews', False)
        augment_passages: bool = config.get('augment_passages', False)

        if augment_passages and augment_reviews or (not augment_passages and not augment_reviews):
            # TODO: Replace with a warning instead of completely prohibiting both being set
            raise NotImplementedError(
                'One of Reviews and Passages should be augmented, but not both.'
            )

        # An empty dict in .yml is valid, cause this augmenter can take no args. If it doesn't exist, it should
        #  NOT be added, and should be set to None. This is why extract_augmenter_args is abstracted into its own fn
        has_cwe, augment_context_embedding = extract_augmenter_args(config, 'augment_context_embedding')
        has_lambada, augment_lambada = extract_augmenter_args(config, 'augment_lambada')
        model_directory: str = config.get('lambada_model_directory', None)
        has_abs_sum, augment_abstractive_summarization = extract_augmenter_args(config,
                                                                                  'augment_abstractive_summarization')
        has_rand, augment_random_behavior = extract_augmenter_args(config, 'augment_random_behavior')

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
        AugDataPipeline.nlpaug_config = NLPAugConfig(augmenter_pipeline, augment_passages, augment_reviews)

    @staticmethod
    def tokenizer_factory(_tok: Callable, encoder: BaseEncoder) -> Callable:
        """Function factory that creates a collate function for use with a torch.util.data.Dataloader

        Args:
            tokenizer (PreTrainedTokenizer): A Huggingface model tokenizer, taking strings to torch Tensors
            context_len (int): Max length of the passages passed to the tokenizer

        Returns:
            Callable: A function that will take a batch of string tuples and tokenize them properly.
        """
        # TODO: How can I access this?
        pipeline_config: NLPAugConfig = AugDataPipeline.nlpaug_config

        @typechecked
        def collate(
                data: Iterable[Tuple[str, str]]
        ) -> Tuple[BatchElement, BatchElement]:

            passages, reviews = zip(*data)
            pass_tokens, rev_tokens = _tok(list(passages)), _tok(list(reviews))
            pass_masks = pass_tokens["attention_mask"]
            rev_masks = rev_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]
            rev_tokens = rev_tokens["input_ids"]

            new_rev_tokens, new_rev_masks = [*rev_tokens], [*rev_masks]
            new_pass_tokens, new_pass_masks = [*pass_tokens], [*pass_masks]
            # Augment pass_tokens, rev_tokens with the Sequential() instance. TODO: None of this is correct. Probably
            if pipeline_config.augment_reviews:
                augmented_data = pipeline_config.augmenter_sequence.augment(rev_tokens)
                multiple = len(augmented_data)
                # [a, b..z] --> [a, b..z, a+, b+...z+]
                [new_rev_tokens.extend(augmentation) for augmentation in augmented_data]
                [new_rev_masks.extend(rev_masks) for _ in augmented_data]
                new_pass_tokens, new_pass_masks = new_pass_tokens * multiple, new_pass_masks * multiple

            if pipeline_config.augment_passages:
                augmented_data = pipeline_config.augmenter_sequence.augment(pass_tokens)
                multiple = len(augmented_data)
                # [a, b..z] --> [a, b..z, a+, b+...z+]
                [new_pass_tokens.extend(augmentation) for augmentation in augmented_data]
                [new_pass_masks.extend(rev_masks) for _ in augmented_data]
                new_pass_tokens, new_pass_masks = new_pass_tokens * multiple, new_pass_masks * multiple

            # TODO: Augmented passages should be paired with their reviews, augmented reviews should be paired with
            #  their passages. Then, either/both should be appended to the tokens/masks? and returned as BatchElements?
            return (
                BatchElement(new_pass_tokens, new_pass_masks),
                BatchElement(new_rev_tokens, new_rev_masks),
            )

        return collate
