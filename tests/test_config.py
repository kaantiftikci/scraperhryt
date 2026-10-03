"""Settings varsayılanları ve türetilmiş listeler."""

from __future__ import annotations

from scraperhryt.config import Settings


def test_csv_helpers_strip_and_skip_blanks() -> None:
    s = Settings(_env_file=None, keywords=" bakan, ,fon ", sources="hurriyet,,12punto")
    assert s.keyword_list == ["bakan", "fon"] and s.source_list == ["hurriyet", "12punto"]


def test_default_punto_categories_cover_all_site_sections() -> None:
    """Karışık /rss yalnızca 20 öğe verir; her bölümün kendi /rss/<kategori> beslemesi listede olmalı."""
    cats = Settings(_env_file=None).punto_category_list
    assert len(cats) == len(set(cats))
    assert {
        "gundem", "siyaset", "ekonomi", "turkiye", "kamu-gundemi", "is-dunyasi", "secim",
        "otomotiv", "seyahat", "gurme", "trend-bilgi-kapsulu",
    } <= set(cats)
    assert "24saat" not in cats  # /24saat karışık beslemenin aynısıdır, ayrı bölüm değil
