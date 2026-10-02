#!/usr/bin/env bash
# Verilen host:port adresleri TCP düzeyinde erişilebilir olana kadar bekler, ardından (verildiyse) komutu çalıştırır.
#
#   scripts/wait-for.sh [-t SANİYE] host:port [host:port ...] [-- komut [arg ...]]
#
# Örnek: scripts/wait-for.sh -t 120 localhost:5672 localhost:9200 -- scraperhryt run-all
# Çıkış kodu: hedeflerden biri süre içinde açılmazsa 1; komut verildiyse komutun çıkış kodu.
set -euo pipefail

usage() {
  sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
}

wait_seconds=60
targets=()
while [ $# -gt 0 ]; do
  case "$1" in
    -t|--timeout)
      [ $# -ge 2 ] || { echo "wait-for: $1 için değer gerekli" >&2; exit 2; }
      wait_seconds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      targets+=("$1")
      shift
      ;;
  esac
done

if [ ${#targets[@]} -eq 0 ]; then
  usage >&2
  exit 2
fi

probe() {
  # /dev/tcp bash'in yerleşik TCP istemcisidir; coreutils `timeout` varsa bağlantı denemesi 3 sn ile sınırlanır.
  if command -v timeout >/dev/null 2>&1; then
    timeout 3 bash -c "exec 3<>/dev/tcp/$1/$2" >/dev/null 2>&1
  else
    bash -c "exec 3<>/dev/tcp/$1/$2" >/dev/null 2>&1
  fi
}

wait_one() {
  local target="$1" host port deadline
  host="${target%:*}"
  port="${target##*:}"
  if [ -z "$host" ] || [ -z "$port" ] || [ "$host" = "$target" ]; then
    echo "wait-for: geçersiz hedef '$target' (host:port bekleniyor)" >&2
    return 2
  fi
  deadline=$(( $(date +%s) + wait_seconds ))
  until probe "$host" "$port"; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "wait-for: $target ${wait_seconds}s içinde erişilebilir olmadı" >&2
      return 1
    fi
    sleep 1
  done
  echo "wait-for: $target hazır" >&2
}

for target in "${targets[@]}"; do
  wait_one "$target"
done

if [ $# -gt 0 ]; then
  exec "$@"
fi
