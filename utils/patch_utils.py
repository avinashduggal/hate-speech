import os
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"

import tensorflow_hub as hub

import textattack.constraints.semantics.sentence_encoders.sentence_encoder as se



def apply_sentence_encoder_patch():
    # Load TF-hub library locally, due to issues with the pip version.
    os.environ["TFHUB_CACHE_DIR"] = "../tfhub_cache"

    print("Starting download/load of Universal Sentence Encoder...")
    
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("USE loaded successfully")

    original_score_list = se.SentenceEncoder._score_list

    """
        Monkey-Patched version of score-list. 
        This is to prevent StopIteration errors from stopping the adversarial training prematurely.
        If StopIteration error occurs, then just return 0 to filter out the candidate text.
    """
    def patched_score_list(self, reference_text, transformed_texts):
        if len(transformed_texts) == 0:
            return []
        try:
            scores = original_score_list(self, reference_text, transformed_texts)
            if scores is None or len(scores) == 0:
                return [0.0] * len(transformed_texts)
            return scores
        except StopIteration:
            return [0.0] * len(transformed_texts)

    se.SentenceEncoder._score_list = patched_score_list

    original_check_constraint_many = se.SentenceEncoder._check_constraint_many

    """
        Monkey-patch version of check constraints many. 
        Like the original method, it will filter the list of transformed texts, 
        so that the similarity between reference text and transformed text is greater than the threshold.
        New functionality includes setting similarity scores to zeros if StopIteration error occurs.
        Fix score length mismatch by padding missing scores to zeros or trimming excess scores.
    """
    def patched_check_constraint_many(self, transformed_texts, reference_text):
        if len(transformed_texts) == 0:
            return []

        try:
            scores = self._score_list(reference_text, transformed_texts)
        except StopIteration:
            scores = [0.0] * len(transformed_texts)

        if hasattr(scores, "numpy"):
            scores = scores.numpy()
        scores = list(scores) if not isinstance(scores, list) else scores

        if len(scores) == 0:
            scores = [0.0] * len(transformed_texts)
        elif len(scores) < len(transformed_texts):
            scores = scores + [0.0] * (len(transformed_texts) - len(scores))
        elif len(scores) > len(transformed_texts):
            scores = scores[:len(transformed_texts)]

        result = []
        for i, transformed_text in enumerate(transformed_texts):
            try:
                score = scores[i].item() if hasattr(scores[i], "item") else float(scores[i])
                transformed_text.attack_attrs["similarity_score"] = score
                if score >= self.threshold:
                    result.append(transformed_text)
            except (IndexError, AttributeError):
                continue

        return result

    se.SentenceEncoder._check_constraint_many = patched_check_constraint_many