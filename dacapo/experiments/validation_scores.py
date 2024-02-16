"""
This module defines the class ValidationScores and it's associated methods. It is used to 
validate the dataset on the basis of evaluation scores and post processing parameters.

Classes:
    ValidationScores: Class for handling, managing and retrieving validation scores.

The module makes use of the following packages:
- attr for defining classes
- numpy for numerical operations
- xarray for labeled data functionalities 
"""

@attr.s
class ValidationScores:
    """
    Class for handling, managing and retrieving validation scores.
    
    Attributes:
        parameters (List[PostProcessorParameters]): List of parameters that will be evaluated.
        datasets (List[Dataset]): List of datasets that will be evaluated at each iteration.
        evaluation_scores (EvaluationScores): The scores that are collected on each iteration per PostProcessorParameters and Dataset.
        scores (List[ValidationIterationScores]): A list of evaluation scores and their associated post-processing parameters.
    """
    parameters: List[PostProcessorParameters] = attr.ib(
        metadata={"help_text": "The list of parameters that are being evaluated"}
    )
    datasets: List[Dataset] = attr.ib(
        metadata={"help_text": "The datasets that will be evaluated at each iteration"}
    )
    evaluation_scores: EvaluationScores = attr.ib(
        metadata={
            "help_text": "The scores that are collected on each iteration per "
            "`PostProcessorParameters` and `Dataset`"
        }
    )
    scores: List[ValidationIterationScores] = attr.ib(
        factory=lambda: list(),
        metadata={
            "help_text": "A list of evaluation scores and their associated post-processing parameters."
        },
    )

    def subscores(
        self, iteration_scores: List[ValidationIterationScores]
    ) -> "ValidationScores":
        """
        Sub-function for ValidationScores.
        
        Args:
            iteration_scores (List[ValidationIterationScores]): List of iteration scores.
            
        Returns:
            ValidationScores object with updated iteration scores.
        """

        return ValidationScores(
            self.parameters,
            self.datasets,
            self.evaluation_scores,
            scores=iteration_scores,
        )

    def add_iteration_scores(
        self,
        iteration_scores: ValidationIterationScores,
    ) -> None:
        """
        Appends more iteration scores to the existing list of scores.
        
        Args:
            iteration_scores (ValidationIterationScores): New iteration scores.
        """
        
        self.scores.append(iteration_scores)

    def delete_after(self, iteration: int) -> None:
        """
        Deletes the scores for the iterations after the given iteration number.
        
        Args:
            iteration (int): The iteration number after which scores will be deleted.
        """
        
        self.scores = [scores for scores in self.scores if scores.iteration < iteration]

    def validated_until(self) -> int:
        """
        Determines the number of iterations that the validation has been performed for.
        
        Returns:
            An integer denoting the number of iterations validated (the maximum iteration plus one)
        """
        
        if not self.scores:
            return 0
        return max([score.iteration for score in self.scores]) + 1

    def compare(self, existing_iteration_scores: List[ValidationIterationScores]) -> Tuple[bool, int]:
        """
        Compares iteration stats provided from elsewhere to scores we have saved locally. Local 
        scores take priority. If local scores are at a lower iteration than the existing ones, 
        delete the existing ones and replace with local. If local iteration > existing iteration, 
        just update existing scores with the last overhanging local scores.
        
        Args:
            existing_iteration_scores (List[ValidationIterationScores]): List of existing iteration scores.

        Returns:
            A tuple containing a boolean indicating whether the existing iteration is above the 
            current iteration, and the number of the existing iteration.
        """

    @property
    def criteria(self) -> List[str]:
        """
        Property for returning the evaluation criteria used.
        
        Returns:
            A list of parameters that were used as evaluation criteria.
        """
        
        return self.evaluation_scores.criteria

    @property
    def parameter_names(self) -> List[str]:
        """
        Property for returning the names of the parameters.
        
        Returns:
            A list of names of the parameters.
        """
        
        return self.parameters[0].parameter_names

    def to_xarray(self) -> xr.DataArray:
        """
        Returns a xarray object containing iteration score information.
        
        Returns:
            xarray data array containing the iteration scores, reshaped in accordance with the
            datasets, parameters and criteria.
        """

    def get_best(self, data: xr.DataArray, dim: str) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute the Best scores along dimension "dim" per criterion. Returns both the index 
        associated with the best value, and the best value in two seperate arrays.
        
        Args:
            data (xarray DataArray): Contains the iteration data from which the best parameters will be computed.
            dim (str): The dimension along which to carry out the computation.

        Returns:
            Two xarray DataArrays, one containing the best indexes and the other containing the best scores.
        """
