# RAG-5060 на том же Windows, что и координатор fastagg

Координатор fastagg слушает совместимый порт `127.0.0.1:11435`. RAG в WSL2 ходит туда вместо контейнера `ollama`. Исходники RAG этот каталог не меняет: разработчик RAG подключает `EMBEDDING_URL` и `RERANK_URL` сам. Ниже один фрагмент, как такой вызов выглядит.

Полный контракт: [API_CONTRACT.md](API_CONTRACT.md).

## Сеть WSL

Два режима WSL2.

В режиме mirrored `127.0.0.1` внутри WSL — это localhost Windows, и контейнер с `network_mode: host` видит порт координатора напрямую. У RAG сеть мостовая, поэтому этот путь не используется.

В режиме NAT (обычный) контейнер видит Windows как `host.docker.internal`. Override добавляет `extra_hosts: host.docker.internal:host-gateway`. Адрес Windows из WSL (`ip route` / `cat /etc/resolv.conf`, обычно vEthernet (WSL)) нужен только если host-gateway не резолвится: подставьте его вместо `host.docker.internal`.

Порт `11435` по умолчанию слушает только `127.0.0.1`. Для WSL этого достаточно: host-gateway попадает в loopback Windows. `--api-bind 0.0.0.0` нужен, если проверка `curl` с хоста WSL на LAN-IP Windows не доходит до localhost. Тогда откройте TCP `11435` в брандмауэре для профиля, в котором сидит vEthernet (WSL). Изолированная сеть часто помечается Public, поэтому правило нужно и для Private, и для Public. Правило установщик сам не создаёт.

## Запуск без контейнера Ollama

Из каталога RAG-5060, файл override скопирован рядом или указан полным путём:

```text
docker compose -f docker-compose.yml -f docker-compose.fastagg.yml --profile core up -d
docker compose -f docker-compose.yml -f docker-compose.fastagg.yml --profile core config --services
```

Во втором списке не должно быть `ollama`. Сервис `api` больше не ждёт его healthcheck: в override зависимости перечислены заново (`depends_on: !override`), иначе Compose склеит старый список с новым и снова потребует Ollama.

Освободить видеопамять, если контейнер уже был запущен:

```text
docker compose stop ollama
```

Затем поднимите модель в дашборде fastagg. На одной RTX 5060 Ti контейнер Ollama и кластерная модель вместе не живут.

Проверка из контейнера `api` (схема собирается в команде, чтобы не путать с адресом дашборда):

```text
docker compose exec api sh -c 'curl -fsS "$OLLAMA_HOST/api/tags" && curl -fsS "$OLLAMA_HOST/api/health"'
```

Ожидаются имена `fastagg` и `qwen3:8b` в tags и JSON health с полями `roles.llm`, `roles.embedding`, `roles.reranker`.

## Вернуть локальный Ollama

Один профиль, тот же override:

```text
docker compose -f docker-compose.yml -f docker-compose.fastagg.yml --profile core --profile ollama up -d
```

Образ `ollama/ollama:0.12.6` и том `ollamadata` остаются в базовом файле. Перед этим остановите модель fastagg, иначе обе службы претендуют на GPU. В `.env` на время отката верните `OLLAMA_HOST=http://ollama:11434`.

## Фрагмент для клиента RAG

Это не патч репозитория RAG. Так клиент может взять эмбеддинги и реранкер с координатора, если переменные заданы, и остаться на прежнем Ollama для чата, пока `OLLAMA_HOST` смотрит на fastagg.

```python
import os
import httpx

host = os.environ["OLLAMA_HOST"].rstrip("/")
token = os.environ.get("FASTAGG_TOKEN") or ""
headers = {"Authorization": f"Bearer {token}"} if token else {}

def embed(texts: list[str]) -> list[list[float]]:
    url = os.environ.get("EMBEDDING_URL") or f"{host}/v1/embeddings"
    response = httpx.post(url, json={"input": texts}, headers=headers, timeout=30)
    response.raise_for_status()
    body = response.json()
    # embedding_model_id стабилен (bge-m3-Q8_0); dim 1024; векторы уже L2 ~ 1
    return [item["embedding"] for item in body["data"]]

def rerank(query: str, documents: list[str], top_n: int = 5) -> list[dict]:
    url = os.environ.get("RERANK_URL") or f"{host}/v1/rerank"
    response = httpx.post(
        url,
        json={"query": query, "documents": documents, "top_n": top_n},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["results"]  # [{index, relevance_score}]
```

Чат RAG уже ходит в `POST {OLLAMA_HOST}/api/generate` с `options.num_ctx` и `think`. Ответ fastagg кладёт эффективный контекст в `effective_ctx` и заголовок `X-Fastagg-Effective-Ctx`. При `429` смотрите `Retry-After` (2 секунды). При `503` на `/api/health` поле `roles.*.replacing` показывает, идёт ли перенос.
