"""Settings varsayılanları ve türetilmiş listeler."""

from __future__ import annotations

from scraperhryt.config import Settings


def test_csv_helpers_strip_and_skip_blanks() -> None:
    s = Settings(_env_file=None, keywords=" bakan, ,fon ", sources="hurriyet,,12punto")
    assert s.keyword_list == ["bakan", "fon"] and s.source_list == ["hurriyet", "12punto"]


def test_default_punto_categories_cover_news_sections_only() -> None:
    """Karışık /rss yalnızca 20 öğe verir; gündemle ilgili her bölümün kendi beslemesi listede olmalı."""
    cats = Settings(_env_file=None).punto_category_list
    assert len(cats) == len(set(cats))
    assert {"gundem", "siyaset", "ekonomi", "turkiye", "kamu-gundemi", "is-dunyasi", "secim", "adalet-hukuk"} <= set(cats)
    assert not {"spor", "otomotiv", "seyahat", "gurme", "yasam", "kultur-sanat", "trend-bilgi-kapsulu"} & set(cats)
    assert "24saat" not in cats  # /24saat karışık beslemenin aynısıdır, ayrı bölüm değil
