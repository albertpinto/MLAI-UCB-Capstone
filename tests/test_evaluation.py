

def test_corpus_of_keeps_only_the_chosen_documents():
    from capstone.evaluation import Corpus, corpus_of

    corpus = Corpus(["a", "b", "c"], ["d1:0", "d1:1", "d2:0"],
                    {"d1:0": {"section": "A"}, "d1:1": {"section": "B"}, "d2:0": {"section": "C"}}, [("d1", 0), ("d1", 1), ("d2", 0)])
    only = corpus_of(corpus, {"d1"})
    assert only.texts == ["a", "b"] and only.keys == ["d1:0", "d1:1"] and set(only.sections) == {"d1:0", "d1:1"} and only.ids == [("d1", 0), ("d1", 1)]
