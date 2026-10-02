"""Raporlama katmanı, RAG soru-cevap ve HTTP API.

- ``prompts``: Türkçe istemler (rapor anlatısı, sorgu yeniden yazma, RAG yanıtı) ve ortak biçimleme yardımcıları
- ``builder``: ``ReportBuilder`` — pencere istatistikleri + en yüksek alarmlar + LLM anlatısı (şablon yedekli)
- ``rag``: ``QAEngine`` — sorgu yeniden yazma → hibrit arama (BM25 + isteğe bağlı kNN, RRF) → en yeni önce
  bağlam → atıflı Türkçe yanıt
- ``service``: ``ReportingConsumer`` (q.alarms → alarm özetleri) ve ``PeriodicReporter`` (periyodik raporlar)
- ``api``: ``create_app`` — FastAPI uç noktaları ve Jinja2 panosu (``templates/dashboard.html``)
"""
