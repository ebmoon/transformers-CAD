from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    pass

class CheckerState:
    """Abstract base class for all checker states"""

class Checker:
    """Abstract base class for all checkers that can be applied during checker guided generation."""

    def filter_vocab(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Filter out next tokens for the current input that do not pass the checker.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, num_accepted_tokens)` containing the accepted tokens
            for each batch.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `filter_vocab`."
        )


    def update(self, next_tokens: torch.LongTensor) -> CheckerState:
        """
        Update the state of the checker based on the selected next tokens.

        Args:
            next_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of selected next tokens in the vocabulary.

        Return:
            `CheckerState` after updating the state.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `update`."
        )