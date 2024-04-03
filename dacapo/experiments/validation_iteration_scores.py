from typing import List
import attr


@attr.s
class ValidationIterationScores:
    """
    A class used to represent the validation iteration scores in an organized structure.

    Attributes:
        iteration (int): The iteration associated with these validation scores.
        scores (List[List[List[float]]]): A list of scores per dataset, post processor
        parameters, and evaluation criterion.
    Note:
        The scores list is structured as follows:
        - The outer list contains the scores for each dataset.
        - The middle list contains the scores for each post processor parameter.
        - The inner list contains the scores for each evaluation criterion.

    """

    iteration: int = attr.ib(
        metadata={"help_text": "The iteration associated with these validation scores."}
    )
    scores: List[List[List[float]]] = attr.ib(
        metadata={
            "help_text": "A list of scores per dataset, post processor "
            "parameters, and evaluation criterion."
        }
    )
