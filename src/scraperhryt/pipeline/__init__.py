"""Boru hattı aşamaları: anahtar kelime filtresi, Ollama LLM skorlama ve alarm katmanı.

- ``keyword_filter``: q.articles.raw → (eşleşme) q.articles.keyword / (eşleşme yok) q.articles.scored
- ``llm``: Ollama HTTP istemcisi, test için FakeOllama ve çevrimdışı HeuristicLLM
- ``prompts``: Türkçe analist istemleri ve alarm rubriği
- ``scorer``: q.articles.keyword → LLM kararı → q.articles.scored
- ``alarm``: q.articles.scored → Elasticsearch + alarm kanalları (ayrı modül)
"""
