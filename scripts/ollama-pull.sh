#!/bin/sh
# OLLAMA_MODEL (ve doluysa OLLAMA_EMBEDDING_MODEL) modellerini Ollama sunucusuna indirir; yüklü olanları atlar.
#
#   docker compose --profile ollama up      → ollama-pull servisi bu betiği konteynerde çalıştırır
#   OLLAMA_HOST=http://localhost:11434 scripts/ollama-pull.sh   → ana makinede (ollama CLI kurulu olmalı)
#
# Ortam: OLLAMA_HOST (varsayılan http://localhost:11434), OLLAMA_MODEL (qwen2.5:7b), OLLAMA_EMBEDDING_MODEL (boş),
#        WAIT_SECONDS (sunucuyu bekleme süresi, 180).
set -eu

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
OLLAMA_EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL:-}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
export OLLAMA_HOST

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama-pull: 'ollama' komutu bulunamadı (https://ollama.com/download)" >&2
  exit 127
fi

echo "ollama-pull: sunucu bekleniyor ($OLLAMA_HOST, en çok ${WAIT_SECONDS}s)..."
elapsed=0
until ollama list >/dev/null 2>&1; do
  if [ "$elapsed" -ge "$WAIT_SECONDS" ]; then
    echo "ollama-pull: $OLLAMA_HOST ${WAIT_SECONDS}s içinde yanıt vermedi" >&2
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

for model in "$OLLAMA_MODEL" $OLLAMA_EMBEDDING_MODEL; do
  [ -n "$model" ] || continue
  # `ollama list` adları her zaman etiketli yazar (nomic-embed-text → nomic-embed-text:latest); etiketsiz adı
  # aynı kurala göre tamamla ki yüklü model yeniden indirilmesin (pipeline/llm.py normalize_model_name ile aynı).
  case "$model" in
    *:*) installed_name="$model" ;;
    *) installed_name="$model:latest" ;;
  esac
  if ollama list | awk 'NR > 1 { print $1 }' | grep -qx -- "$installed_name"; then
    echo "ollama-pull: $model zaten yüklü ($installed_name)"
    continue
  fi
  echo "ollama-pull: indiriliyor: $model"
  ollama pull "$model"
done
echo "ollama-pull: tamam ($OLLAMA_MODEL${OLLAMA_EMBEDDING_MODEL:+, $OLLAMA_EMBEDDING_MODEL})"
