from textattack.attack_recipes import AttackRecipe
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import CompositeTransformation, WordSwapMaskedLM, WordSwapEmbedding
from textattack.goal_functions import UntargetedClassification
from textattack import Attack

"""
    Hybrid AttackRecipe to train DeBERTa-v3 model.
    This combines word and semantic-level attacks in one recipe.
    WordSwapMaskedLM generates word replacements using a MLM model based on roberta-base.
    BERT-Attack performs replacement at a token-level with a maximum of 10 candidates tokens to perform replacements based on MLM's confidence.
    Also performs replacement of words using WordSwapEmbeddings with a maximum of 10 candidate words to replace with synonyms.
    Attack does not change the same candidate token twice and stopwords.
    UniversalSentenceEncoder is to ensure the semantic meaning is preserved while trying to flip the label.
    Uses GreedyWordSwapWIR to remove words based on the importance of a word. 
"""
class DeBERTaAttack(AttackRecipe):
    @staticmethod
    def build(model_wrapper): 
        goal_function = UntargetedClassification(model_wrapper, query_budget=100)

        transformation = CompositeTransformation([
            WordSwapMaskedLM(method="bert-attack", masked_language_model="roberta-base", max_candidates=10),
            WordSwapEmbedding(max_candidates=10),                                                              
        ])

        constraints = [
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