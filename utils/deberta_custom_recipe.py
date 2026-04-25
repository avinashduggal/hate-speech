from textattack.attack_recipes import AttackRecipe
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import CompositeTransformation, WordSwapEmbedding, WordSwapNeighboringCharacterSwap, WordSwapRandomCharacterSubstitution, WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion
from textattack.goal_functions import UntargetedClassification
from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance

"""
    Hybrid AttackRecipe to train DeBERTa-v3 model.
    This combines character and word attacks in one recipe.
    Performs replacement of words using WordSwapEmbeddings with a maximum of 10 candidate words to replace with synonyms.
    WordSwapNeighboringCharacterSwap swaps characters using a neighboring character at target word.
    WordSwapRandomCharacterSubstitution replaces characters at target word.
    WordSwapRandomCharacterDeletion deletes characters at target word.
    WordSwapRandomCharacterInsertion inserts characters at target word.
    LevenshteinEditDistance set to 15 for more subtle perturbations.
    Attack does not change the same candidate token twice and stopwords.
    UniversalSentenceEncoder is to ensure the semantic meaning is preserved while trying to flip the label.
    Uses GreedyWordSwapWIR to remove words based on the importance of a word. 
"""
class DeBERTaAttack(AttackRecipe):
    @staticmethod
    def build(model_wrapper): 
        goal_function = UntargetedClassification(model_wrapper, query_budget=100)

        transformation = CompositeTransformation([
            WordSwapEmbedding(max_candidates=10),                                                              
            WordSwapNeighboringCharacterSwap(),
            WordSwapRandomCharacterSubstitution(),
            WordSwapRandomCharacterDeletion(),
            WordSwapRandomCharacterInsertion(),                                                             
        ])

        constraints = [
            LevenshteinEditDistance(15),
            RepeatModification(),
            StopwordModification(),
            UniversalSentenceEncoder(
                threshold=0.75,
                metric="cosine",
                compare_against_original=True,
                window_size=15
            )
        ]

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)    