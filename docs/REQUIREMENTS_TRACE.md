# scraperhryt — Gereksinim İzleme Matrisi

Bu belge, kullanıcının isteğindeki her maddeyi onu karşılayan **kod yoluna** (`dosya:fonksiyon`) ve bunu
**kanıtlayan teste** bağlar. Test komutları:

```bash
# çevrimdışı (fixtures + bellek içi broker/depo + FakeOllama/HeuristicLLM)
python -m pytest -m "not live" -p no:cacheprovider
# canlı (gerçek sitelere istek atar)
python -m pytest -m live -p no:cacheprovider
# uçtan uca canlı çalıştırma (gerçek kazıma + bellek içi boru hattı + RAG sorusu)
MAX_ARTICLES_PER_RUN=40 scraperhryt run-all --once --in-memory --fake-llm --ask "Fon soruşturmasında son durum ne?"
```

Tüm dosya yolları `src/scraperhryt/` ve `tests/` köküne göredir.

---

## G1. Hürriyet Gündem ve 12punto'daki tüm içerikleri çek

| Alt gereksinim | Kod yolu | Kanıtlayan test |
|---|---|---|
| Hürriyet Gündem keşfi: RSS (100 haber, tam metin) + `/gundem/` liste sayfası; yalnızca `/gundem/` URL'leri | `scrapers/hurriyet.py:HurriyetSource.discover`, `.parse_rss`, `.parse_listing`, `.is_gundem_url` | `test_scrapers.py::test_hurriyet_rss_yields_100_links_with_full_text`, `::test_hurriyet_listing_yields_30_links`, `::test_hurriyet_discover_only_gundem_deduplicated`, `::test_hurriyet_url_filter` |
| Hürriyet haber sayfası ayrıştırma (JSON-LD `NewsArticle`, `h1`, spot, tarih, gövde) ve Google kalıp temizliği | `scrapers/hurriyet.py:HurriyetSource.parse_article`, `.clean_article_body`; `scrapers/base.py:parse_jsonld_newsarticle`, `parse_tr_date` | `test_scrapers.py::test_hurriyet_article_parse`, `::test_hurriyet_article_body_fallback_strips_boilerplate`, `::test_parse_jsonld_handles_dict_list_and_graph`, `::test_parse_tr_date` |
| Sayfa alınamazsa RSS `<text>` + `<abstract>` ile yine kayıt üret | `scrapers/hurriyet.py:HurriyetSource.fetch_article`, `.record_from_rss` | `test_scrapers.py::test_hurriyet_fetch_falls_back_to_rss_on_http_error`, `::test_hurriyet_fetch_falls_back_to_rss_on_timeout`, `::test_hurriyet_fetch_raises_without_rss_fallback` |
| `robots.txt`'e uyum: `/api/` ve `/arama/` yollarına istek yok | `scrapers/hurriyet.py:HurriyetSource.discover` (yalnızca RSS + liste; arşiv/arama yolu yok) | `test_scrapers.py::test_hurriyet_discover_only_gundem_deduplicated` (çağrılan URL'ler fixture ile sınırlı) |
| 12punto keşfi: `/rss` + her kategori için `/rss/<kategori>` + liste sayfaları; `http://` → `https://`; haber olmayan URL'ler atlanır | `scrapers/punto.py:PuntoSource.discover`, `.parse_rss`, `.parse_listing`, `.normalize_url`; `config.py:Settings.punto_category_list` | `test_scrapers.py::test_punto_rss_yields_20_https_links`, `::test_punto_listing_yields_25_links`, `::test_punto_discover_collects_feeds_listings_and_skips_missing`, `::test_punto_url_filter`; `test_config.py::test_default_punto_categories_cover_all_site_sections` |
| 12punto geriye dönük tarama (arşiv araması, gün gün) | `scrapers/punto.py:PuntoSource.discover` (`backfill_days > 0`), `.archive_url`, `.parse_search_page` | `test_scrapers.py::test_punto_archive_url_uses_site_date_format_and_paging`, `::test_punto_discover_backfill_scans_every_day_despite_empty_recent_days`, `::test_punto_discover_backfill_skips_day_on_http_error`; canlı: `test_live_scrapers.py::test_live_punto_archive_search_returns_results_for_indexed_day` |
| 12punto haber sayfası ayrıştırma (JSON-LD listesi, `section.details`, "Yayınlanma:" tarihi) | `scrapers/punto.py:PuntoSource.parse_article` | `test_scrapers.py::test_punto_article_parse`, `::test_punto_article_falls_back_to_jsonld_body_and_text_date` |
| Kibar istemci: UA, zaman aşımı, 5xx/bağlantı hatasında yeniden deneme, site başına bekleme | `scrapers/base.py:HttpClient.get_text` | `test_scrapers.py::test_http_client_retries_transient_errors`, `::test_http_client_does_not_retry_client_errors`, `::test_http_client_politeness_delay_per_host` |
| Sürekli tarama; tekrar kazımayı önleme; içerik değişince yeniden yayınlama; tur başına haber bütçesi | `scrapers/runner.py:ScrapeRunner.run_once`, `.run_forever`; `scrapers/state.py:SeenStore.status`, `.mark` | `test_scrapers.py::test_runner_publishes_raw_records_then_nothing_on_second_run`, `::test_runner_republishes_updated_content`, `::test_runner_respects_total_article_budget`, `::test_run_forever_stops_on_event`, `::test_seen_store_status_transitions` |
| Her haber `NewsRecord` olarak `article.raw` ile `q.articles.raw`'a yayınlanır | `scrapers/runner.py:ScrapeRunner.run_once` (`record.to_message()`, `RoutingKey.ARTICLE_RAW`) | `test_scrapers.py::test_runner_publishes_raw_records_then_nothing_on_second_run`; `test_e2e_inmemory.py::test_e2e_routing_and_alarm_policy` |
| Gerçek siteler hâlâ bu yapıda (canlı doğrulama) | `scrapers/hurriyet.py`, `scrapers/punto.py` | `test_live_scrapers.py::test_live_discover_and_fetch_two_articles` (`-m live`) |

## G2. "bakan, cumhurbaşkanı, fon" gibi anahtar kelime içeren içerikler RabbitMQ kuyruğuna obje olarak gitsin

| Alt gereksinim | Kod yolu | Kanıtlayan test |
|---|---|---|
| Anahtar kelimeler ortamdan (`KEYWORDS`) okunur; Türkçe ek-toleranslı kök eşleşmesi (`bakan` → bakanı, bakanlık…), `=kelime` tam, `re:` regex | `config.py:Settings.keyword_list`; `textutil.py:KeywordMatcher.find`, `.matches`, `tr_lower`, `tr_fold` | `test_textutil.py::test_suffix_tolerant_match_accepts_turkish_inflections`, `::test_suffix_tolerant_match_rejects_embedded_or_foreign_tails`, `::test_fon_matches_only_real_inflections`, `::test_exact_substring_and_regex_modes`, `::test_folded_text_matches_accentless_spelling` |
| Eşleşen haber `matched_keywords` ile zenginleşir, `stage=keyword`, `article.keyword` → `q.articles.keyword` | `pipeline/keyword_filter.py:KeywordFilterService.classify`, `.handle` | `test_pipeline.py::test_hit_routes_to_keyword_queue`, `::test_multiple_keywords_and_suffixes` |
| Eşleşmeyen haber LLM'e gitmez: `mark_not_scored()` ile `article.scored` → `q.articles.scored` (ES'e yine yazılır) | `pipeline/keyword_filter.py:KeywordFilterService.handle`; `models.py:NewsRecord.mark_not_scored` | `test_pipeline.py::test_miss_routes_to_scored_queue`; `test_e2e_inmemory.py::test_e2e_routing_and_alarm_policy` |
| İsteğe bağlı: `LLM_SCORE_ALL=1` ile tüm haberler LLM'e gider | `pipeline/keyword_filter.py:KeywordFilterService.handle` | `test_pipeline.py::test_llm_score_all_routes_everything_to_keyword_queue` |
| Bozuk mesaj ölü mektuba (`Reject`) | `pipeline/keyword_filter.py:KeywordFilterService.handle` | `test_pipeline.py::test_invalid_message_is_rejected`, `::test_invalid_message_is_rejected_to_dead_letter` |
| Kuyruk tüketimi `q.articles.raw` | `pipeline/keyword_filter.py:KeywordFilterService.run` | `test_pipeline.py::test_run_consumes_raw_queue` |

## G3. Ollama üzerinde çalışan bir LLM ile yorumlat, sonucu RabbitMQ'ya gönder

| Alt gereksinim | Kod yolu | Kanıtlayan test |
|---|---|---|
| Ollama istemcisi: `/api/chat` JSON modu, `/api/embed`, sağlık ve model denetimi | `pipeline/llm.py:OllamaClient.chat_json`, `.generate_text`, `.embed`, `.health`, `.model_available` | `test_pipeline.py::test_chat_json_sends_expected_payload_and_parses`, `::test_embed_new_api`, `::test_embed_falls_back_to_legacy_endpoint_on_404`, `::test_model_available_and_tags` |
| Türkçe analist istemi ve 0-100 puanlama rubriği; içerik `OLLAMA_MAX_CONTENT_CHARS` ile kısaltılır | `pipeline/prompts.py`; `pipeline/scorer.py:ScoringService.build_prompts` | `test_pipeline.py::test_system_prompt_mentions_rubric_keys_and_topics`, `::test_user_prompt_contains_fields_and_truncates`, `::test_content_is_truncated_in_prompt` |
| Model çıktısı `LLMVerdict`'e doğrulanır (tip dönüşümleri, sınırlama, JSON ayıklama) | `pipeline/scorer.py:build_verdict`, `coerce_score`, `normalize_topics`; `pipeline/llm.py:extract_json_object` | `test_pipeline.py::test_coercions_and_clamping`, `::test_negative_and_float_scores`, `::test_fenced_json_block`, `::test_missing_score_is_bad_output` |
| Bozuk çıktıda süreç içi 2 yeniden deneme, sonra broker gecikmeli yeniden deneme; Ollama erişilemezse `Unavailable` (mesaj kaybı yok) | `pipeline/scorer.py:ScoringService.score_record`, `.score_with_recovery`, `.handle`, `.wait_for_llm` | `test_pipeline.py::test_recovers_after_bad_output`, `::test_always_bad_output_is_retried_then_published_with_fallback_verdict`, `::test_ollama_outage_is_unavailable_not_counted_against_max_attempts`, `::test_run_waits_for_ollama_before_consuming`; `test_broker.py::test_unavailable_keeps_retrying_beyond_max_attempts` |
| Alarm politikası: `alarm_score >= ALARM_THRESHOLD` → `is_alarm`, `alarm_reason` = gerekçe + "LLM Özeti: …"; aksi halde alanlar boş | `models.py:NewsRecord.apply_verdict`; `pipeline/scorer.py:ScoringService.score_record` | `test_pipeline.py::test_alarm_when_score_above_threshold`, `::test_no_alarm_keeps_verdict_but_empties_alarm_fields`, `::test_threshold_boundary_is_inclusive`, `::test_is_alarm_defaults_to_threshold_policy` |
| Skorlanan kayıt `article.scored` ile `q.articles.scored`'a yayınlanır | `pipeline/scorer.py:ScoringService.handle`, `.run` | `test_pipeline.py::test_verdict_feeds_scoring_service`, `::test_end_to_end_raw_to_scored` |
| Ollama'sız geliştirme/test: `FakeOllama`, deterministik `HeuristicLLM` (`--fake-llm`) | `pipeline/llm.py:FakeOllama`, `HeuristicLLM.evaluate`; `cli.py:build_llm` | `test_pipeline.py::test_risky_text_scores_higher_than_plain`, `::test_sports_text_is_penalised`; `test_cli.py::test_build_llm_keeps_ollama_client_unless_fallback_requested`, `::test_run_all_llm_fallback_is_opt_in` |

## G4. Alarm katmanı: alarm olan ve olmayan her kayıt RabbitMQ + Elasticsearch'e

| Alt gereksinim | Kod yolu | Kanıtlayan test |
|---|---|---|
| RabbitMQ topolojisi: `news.topic` exchange, `q.articles.*`, `q.alarms`, `q.reports`, retry ve ölü mektup kuyrukları | `broker.py:QUEUE_BINDINGS`, `RabbitMQBroker.declare_topology`, `InMemoryBroker` | `test_broker.py::test_declare_topology_declares_retry_exchange_and_bounded_reports_queue`, `::test_failure_goes_to_retry_exchange_with_original_routing_key`, `::test_reject_dead_letters_immediately`, `::test_handler_runs_off_connection_thread_while_heartbeats_are_pumped` |
| **Her** kayıt (`stage=alarm`) `news-articles` indeksine yazılır | `pipeline/alarm.py:AlarmService.handle`; `store.py:ElasticsearchStore.index_record`, `InMemoryStore.index_record` | `test_alarm.py::test_non_alarm_record_is_indexed_only`, `::test_low_score_record_is_not_alarm`; `test_store.py::test_index_get_recent_and_reports_roundtrip` |
| Alarm olan kayıt: `AlarmEvent` üretilir, `news-alarms`'a yazılır, kanallara iletilir, `alarm.raised` → `q.alarms` | `pipeline/alarm.py:AlarmService.handle`; `models.py:AlarmEvent.from_record`; `alarm_sinks.py:build_sinks`, `LogSink.send`, `WebhookSink.send`, `TelegramSink.send` | `test_alarm.py::test_alarm_record_is_indexed_notified_and_published`, `::test_log_sink_appends_jsonl_and_logs_warning`, `::test_webhook_sink_posts_text_and_event`, `::test_telegram_sink_sends_html_and_hides_token`, `::test_failing_sink_does_not_prevent_publish` |
| Yeniden teslimde idempotentlik; içerik değişince yeni alarm | `pipeline/alarm.py:AlarmService.handle` (`alarm_id` + `content_hash` denetimi) | `test_alarm.py::test_redelivery_is_idempotent`, `::test_updated_content_raises_a_new_alarm` |
| ES kesintisinde mesaj kaybı yok (`Unavailable`), bozuk mesaj ölü mektuba | `pipeline/alarm.py:AlarmService.handle`, `.run`; `store.py:ElasticsearchStore` | `test_alarm.py::test_es_outage_is_unavailable_and_does_not_exhaust_attempt_budget`, `::test_invalid_message_goes_to_dead_letter`, `::test_run_waits_for_indices_until_stop` |
| Elasticsearch indeks eşlemeleri (Türkçe analizör, `@timestamp`, anahtar alanlar, isteğe bağlı `dense_vector`) | `store.py:build_article_mapping`, `build_alarm_mapping`, `build_report_mapping`, `ElasticsearchStore.ensure_indices` | `test_store.py::test_build_article_mapping_has_turkish_analyzer_and_optional_embedding`, `::test_build_alarm_and_report_mappings`, `::test_es_ensure_indices_creates_each_index_once` |
| Birleşik nesne alanları (content_url, title, subtitle, published_at, content, alarm_score, alarm_reason) uçtan uca korunur | `models.py:NewsRecord`, `.to_message`, `.from_message`, `.to_es_document` | `test_e2e_inmemory.py::test_e2e_store_contents_and_field_roundtrip`, `::test_e2e_routing_and_alarm_policy` |

## G5. Alarmdan sonra raporlama katmanı; bu katmanın mimarisi

| Alt gereksinim | Kod yolu | Kanıtlayan test |
|---|---|---|
| `q.alarms` tüketicisi: alarm biriktirir, `REPORT_DIGEST_EVERY` / `REPORT_DIGEST_MINUTES` koşulunda `alarm_digest` raporu üretir | `reporting/service.py:ReportingConsumer.handle`, `.is_digest_due`, `.flush`, `.run` | `test_reporting.py::test_reporting_consumer_digests_after_n_alarms`, `::test_reporting_consumer_time_based_digest_and_dedupe`, `::test_reporting_consumer_run_emits_time_based_digest_while_idle`, `::test_reporting_consumer_flushes_on_exit_and_rejects_invalid` |
| Yeniden başlatmada özetlenmemiş alarmların kurtarılması; yayın hatasında tamponun korunması | `reporting/service.py:ReportingConsumer.recover_pending`, `.flush` | `test_reporting.py::test_reporting_consumer_recovers_undigested_alarms_on_start`, `::test_reporting_consumer_keeps_buffer_when_publish_fails`, `::test_reporting_consumer_timed_flush_failure_backs_off_and_retries` |
| Periyodik rapor (`REPORT_INTERVAL_MINUTES`, son `REPORT_WINDOW_HOURS`) | `reporting/service.py:PeriodicReporter.run_once`, `.run` | `test_reporting.py::test_periodic_reporter_run_once_publishes`, `::test_periodic_reporter_run_survives_failures`, `::test_periodic_reporter_run_stops_promptly` |
| Rapor kurucu: ES toplulaştırmaları (`stats`), en yüksek alarmlar, Türkçe LLM anlatısı; LLM yoksa şablon anlatı | `reporting/builder.py:ReportBuilder.build`, `build_template_narrative`, `alarm_stats`, `report_id_for`; `store.py:ElasticsearchStore.stats`, `build_stats_aggs` | `test_reporting.py::test_report_builder_builds_stats_top_alarms_and_narrative`, `::test_report_builder_template_fallback_when_llm_fails`, `::test_report_id_is_deterministic_and_alarm_sensitive`; `test_store.py::test_build_stats_aggs_shape`, `::test_es_stats_parses_aggregations` |
| Raporlar `news-reports` indeksine yazılır ve `report.generated` / `report.alarm_digest` ile `q.reports`'a yayınlanır | `reporting/service.py:ReportingConsumer.flush`, `PeriodicReporter.run_once`; `broker.py:QUEUE_BINDINGS` (`report.#`) | `test_reporting.py::test_periodic_reporter_run_once_publishes`, `::test_reporting_consumer_digests_after_n_alarms`; `test_e2e_inmemory.py::test_e2e_digest_and_rag` |
| API ve pano: `/health`, `/articles/search`, `/articles/{id}`, `/alarms`, `/reports`, `POST /reports/generate`, `/stats`, `GET /` | `reporting/api.py:create_app` (`health`, `search_articles`, `get_article`, `list_alarms`, `list_reports`, `generate`, `stats`, `dashboard`), `generate_report`, `ReportPublisher.publish` | `test_reporting.py::test_api_health`, `::test_api_search_returns_alarm_docs`, `::test_api_get_article`, `::test_api_alarms`, `::test_api_generate_and_list_reports`, `::test_api_stats`, `::test_api_dashboard_renders_html`, `::test_api_maps_store_errors_to_503` |
| Mimari belgesi (raporlama katmanı + RAG tasarımı) | `docs/ARCHITECTURE.md` §5–§6 | (belge; CLI/ayar/kuyruk adları bu entegrasyonda kodla karşılaştırıldı) |

## G6. "… arasındaki son durum ne?" sorusuna son içeriklerden yanıt

| Alt gereksinim | Kod yolu | Kanıtlayan test |
|---|---|---|
| Sorgu yeniden yazma (LLM JSON `search_terms`/`entities`), hata/erişimsizlikte soru belirteçlerine düşüş | `reporting/rag.py:QAEngine.rewrite_query`, `.search_queries`, `question_tokens` | `test_reporting.py::test_qa_engine_fallback_when_rewrite_is_bad_json`, `::test_question_tokens_drops_stopwords_and_keeps_names`, `::test_search_queries_are_narrow_and_deduplicated` |
| Getirme: yenilik ağırlıklı Türkçe BM25 (`search_records`, `since`), isteğe bağlı `knn_search` + karşılıklı sıra birleştirme | `reporting/rag.py:QAEngine.retrieve`, `reciprocal_rank_fusion`; `store.py:build_search_query`, `ElasticsearchStore.search_records`, `.knn_search`, `InMemoryStore.search_records` | `test_reporting.py::test_qa_engine_issues_each_search_query_to_store`, `::test_qa_engine_fuses_knn_results_when_embeddings_enabled`, `::test_reciprocal_rank_fusion_merges_and_ranks`; `test_store.py::test_build_search_query_function_score_with_filters`, `::test_search_newer_identical_document_ranks_first`, `::test_search_tolerates_turkish_suffixes_by_prefix` |
| Bağlam en yeniden en eskiye, tarih ve kaynakla; Türkçe yanıt `[1]`, `[2]` atıflarıyla ve "son durum" sıralamasıyla | `reporting/rag.py:QAEngine.build_context`, `.generate_answer`, `newest_first`; `reporting/prompts.py` | `test_reporting.py::test_qa_engine_answers_newest_first_with_citations`, `::test_newest_first_puts_undated_last`, `::test_prompt_helpers_format_turkish` |
| Kanıt yetersizse bunu söyler; LLM yoksa haberlerden derlenmiş (extractive) yanıt | `reporting/rag.py:QAEngine.ask`, `extractive_answer`, `make_snippet` | `test_reporting.py::test_qa_engine_says_insufficient_when_nothing_matches`, `::test_qa_engine_fallback_when_llm_unavailable`, `::test_qa_engine_with_heuristic_llm_is_deterministic` |
| Erişim yolları: `scraperhryt ask "<soru>" [--since-days N] [--top-k N] [--json]`, `POST /ask`, pano soru kutusu, `run-all --once --ask` | `cli.py:cmd_ask`, `print_ask_answer`, `run_all_once_in_memory`; `reporting/api.py:create_app` (`ask`); `reporting/templates/dashboard.html` | `test_cli.py::test_ask_prints_answer_with_sources`, `::test_ask_json_output`, `::test_run_all_once_in_memory_fake_llm`; `test_reporting.py::test_api_ask_returns_citations`, `::test_api_ask_falls_back_without_llm`, `::test_api_ask_marks_no_evidence_distinct_from_llm_outage`; `test_e2e_inmemory.py::test_e2e_digest_and_rag` |

## G7. Çapraz kesen gereksinimler (işletme, dayanıklılık, dağıtım)

| Gereksinim | Kod yolu | Kanıtlayan test |
|---|---|---|
| Tüm katmanlar tek süreçte (`run-all`), her tüketici kendi broker örneğiyle, SIGINT/SIGTERM ile düzgün kapanış | `cli.py:cmd_run_all`, `Supervisor`, `signal_scope`, `consume_loop`, `shutdown_timeout_for` | `test_cli.py::test_run_all_once_rabbitmq_mode_digests_alarms_and_closes_brokers_in_owner_threads`, `::test_run_all_shares_stop_event_with_scorer_and_alarm_services`, `::test_supervisor_records_crash_and_stops_everyone`, `::test_shutdown_timeout_covers_worst_case_llm_call` |
| Altyapısız geliştirme: `--in-memory --fake-llm` ile uçtan uca çalıştırma ve özet | `cli.py:run_all_once_in_memory`, `print_run_summary`; `broker.py:InMemoryBroker`; `store.py:InMemoryStore` | `test_cli.py::test_run_all_once_in_memory_fake_llm`; `test_e2e_inmemory.py` (3 test) |
| Kurulum/denetim: `setup` (topoloji + indeksler + model), `check` (erişilebilirlik) | `cli.py:cmd_setup`, `cmd_check`, `probe_rabbitmq`, `probe_elasticsearch`, `probe_ollama`, `wait_for_ollama` | `test_cli.py::test_setup_fails_when_ollama_is_not_ready`, `::test_setup_ollama_optional_flag_and_env`, `::test_setup_waits_for_ollama_model_when_wait_is_requested`, `::test_check_reports_and_exit_code`, `::test_probes_fail_fast_on_closed_ports` |
| Ayarlar ortam değişkenlerinden; `.env.example` tüm ayarları kapsar | `config.py:Settings`, `get_settings`; `cli.py:build_settings` | `test_cli.py::test_env_example_covers_settings`, `::test_cli_overrides_settings`, `::test_invalid_env_value_is_reported`; `test_config.py::test_csv_helpers_strip_and_skip_blanks` |
| Docker dağıtımı: `docker-compose.yml` (rabbitmq, elasticsearch, kibana/ollama profilleri, setup, scraper, filter, scorer, alarm, reporter, api), `Dockerfile`, `Makefile` | `docker-compose.yml`, `Dockerfile`, `Makefile`, `scripts/wait-for.sh`, `scripts/ollama-pull.sh` | `test_cli.py::test_docker_compose_structure`, `::test_dockerfile_and_makefile`, `::test_ollama_pull_downloads_missing_models` |
| Saat dilimli tarihler (naif → Europe/Istanbul), Türkçe tarih biçimleri | `scrapers/base.py:parse_tr_date`; `cli.py:fmt_dt` | `test_scrapers.py::test_parse_tr_date`, `::test_parse_tr_date_same_instant_across_formats`; `test_cli.py::test_fmt_dt_uses_istanbul_time` |
| Gözlenebilirlik: Türkçe günlükler, kimlik bilgisi gizleme | `logging_setup.py`; `cli.py:redact_url`; `pipeline/llm.py:redact_url` | `test_cli.py::test_logging_configure_is_idempotent`, `::test_redact_url_hides_password`; `test_pipeline.py::test_credentials_in_base_url_are_redacted` |

---

## Son doğrulama (bu entegrasyon turu)

- `ruff check src tests` → temiz.
- `pytest -m "not live"` → **363 geçti**, 3 canlı test dışarıda bırakıldı.
- `pytest -m live` → **3 geçti** (Hürriyet RSS/liste + 12punto RSS/liste/arşiv, gerçek siteler).
- `MAX_ARTICLES_PER_RUN=40 scraperhryt run-all --once --in-memory --fake-llm --ask "Fon soruşturmasında son durum ne?"`
  → 40 haber kazındı (20 Hürriyet + 20 12punto), 13'ü anahtar kelime eşleşti, 40'ı skorlandı/indekslendi,
  4 alarm, 1 alarm özeti + 1 periyodik rapor, ölü mektup 0; soru gerçek haberlerden 4 kaynak atıfıyla yanıtlandı
  (sayılar canlı içeriğe bağlıdır; her çalıştırmada değişir).
- `docker-compose.yml` `yaml.safe_load` ile doğrulandı; README / ARCHITECTURE / RUNBOOK / `.env.example`'daki
  CLI bayrakları, ortam değişkenleri, kuyruk ve routing key adları `cli.py` / `config.py` / `broker.py` ile
  karşılaştırıldı (tek sapma `PUNTO_CATEGORIES` varsayılanıydı; düzeltildi).
