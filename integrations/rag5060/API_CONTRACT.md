# Контракт совместимого порта fastagg для RAG-5060

Порт по умолчанию `127.0.0.1:11435` (`fastagg coordinator --api-bind 127.0.0.1 --api-port 11435`).
Это отдельный процесс от дашборда `47080` и от nginx RAG на `127.0.0.1:8080` внутри WSL.
`--api-bind 0.0.0.0` открывает порт в LAN. Брандмауэр Windows для него установщик сам не создаёт:
правило TCP нужно добавить вручную, и только если bind не localhost.

Если задан `--api-token`, каждый запрос, включая health, требует `Authorization: Bearer <token>`.
Без токена ответ `401` и тело `{"error":"Нужен Bearer-токен fastagg"}`.

Идентичность моделей по умолчанию (файлы проверены в Hugging Face, в этот репозиторий не вшиты):

| Роль | `embedding_model_id` | Файл GGUF | Квант | Размерность |
| --- | --- | --- | --- | --- |
| embedding | `bge-m3-Q8_0` | `bge-m3-Q8_0.gguf` | `Q8_0` | 1024 |
| rerank | `bge-reranker-v2-m3-Q8_0` | `bge-reranker-v2-m3-Q8_0.gguf` | `Q8_0` | нет |

Репозитории: `gpustack/bge-m3-GGUF` и `gpustack/bge-reranker-v2-m3-GGUF`.
Строка `embedding_model_id` стабильна: RAG версионирует по ней кэш векторов и коллекцию Qdrant.
Смена кванта (например на `bge-m3-Q4_K_M`) меняет и эту строку, и имя файла. Координатор сам векторы не перенормирует.

## 1. Эмбеддинги и реранкер

Таймаут прокси на узел роли: 30 с. Таймаут генерации: 300 с, как у клиента RAG.
Пакет эмбеддингов больше 32 строк — `400` и `{"error":"Пакет больше 32"}`.
Длиннее 512 входных токенов координатор не режет: это лимит llama-server роли (`--embedding`, pooling `mean`). Обрезание на стороне RAG.

Нормализация: ответ llama-server `--embedding` для bge-m3 — L2-норма около 1. Координатор вектор не меняет. RAG может проверить `sum(x*x) ~ 1`. Поле `"normalized": true` это описывает, а не обещает повторную нормировку.

Роль не назначена, процесс не поднят или узел не ответил: `503` и `{"error":"..."}`.
Очередь генерации на эмбеддинги и реранкер не распространяется.

### `POST /v1/embeddings`

Запрос, форма OpenAI. `input` — строка или массив строк.

```json
{"model": "bge-m3-Q8_0", "input": ["первый фрагмент", "второй фрагмент"]}
```

Ответ `200`:

```json
{
  "object": "list",
  "model": "bge-m3-Q8_0.gguf",
  "embedding_model_id": "bge-m3-Q8_0",
  "quant": "Q8_0",
  "dim": 1024,
  "normalized": true,
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.01, 0.02]},
    {"object": "embedding", "index": 1, "embedding": [0.03, 0.04]}
  ]
}
```

`model` — точное имя файла GGUF. `embedding_model_id` — стабильный ключ кэша. Длина каждого `embedding` равна `dim` (1024 для bge-m3).

`POST /api/embed` и `POST /api/embeddings` (Ollama) принимают `input` или `prompt` и отвечают:

```json
{
  "model": "bge-m3-Q8_0.gguf",
  "embedding_model_id": "bge-m3-Q8_0",
  "embeddings": [[0.01, 0.02]],
  "dim": 1024
}
```

### `POST /v1/rerank`

Форма llama.cpp: `query`, `documents`, необязательный `top_n`.

```json
{
  "query": "какой фрагмент ближе",
  "documents": ["первый", "второй", "третий"],
  "top_n": 2
}
```

Ответ `200` без обёртки. Индекс — позиция во входном `documents`.

```json
{
  "results": [
    {"index": 1, "relevance_score": 0.91},
    {"index": 0, "relevance_score": 0.42}
  ]
}
```

`embedding_model_id` реранкера в health и в назначении роли: `bge-reranker-v2-m3-Q8_0`, файл `bge-reranker-v2-m3-Q8_0.gguf`.

## 2. Ollama `/api/chat` и `/api/generate`

Оба метода есть. `stream: true` отвечает `application/x-ndjson`: одна строка с уже собранным текстом (`done: false`) и завершающая строка `done: true`. Это не поток токенов по мере генерации. Заголовок `X-Fastagg-Effective-Ctx` стоит на ответе.

Непоточный ответ — один JSON.

Поддерживаются `options.temperature` (если нет — 0.8), `options.num_predict` (если нет — 256), `options.stop` (строка или массив уходит в llama-server как `stop`), `options.num_ctx`.

`num_ctx` больше контекста, с которым поднят кластер: значение зажимается до этого контекста. Эффективный контекст запроса = `min(num_ctx, context_length кластера)`. Поднять контекст выше можно только новым запуском модели на вкладке «Модели». При зажиме `clamped: true`. Если `num_ctx` нет или он ≤ 0, берётся контекст кластера и `clamped: false`.

Заголовок `X-Fastagg-Effective-Ctx` и поля `effective_ctx`, `clamped`, `context_length` сообщают то же число.

`/api/generate`, `stream: false`:

```json
{
  "model": "qwen3:8b",
  "prompt": "вопрос",
  "stream": false,
  "think": false,
  "options": {"num_ctx": 8192, "temperature": 0.2, "num_predict": 256, "stop": ["\n\n"]}
}
```

```json
{
  "model": "qwen3:8b",
  "response": "ответ без блока think",
  "done": true,
  "done_reason": "stop",
  "effective_ctx": 4096,
  "clamped": true,
  "context_length": 4096
}
```

`/api/chat` собирает `messages` в текст `role: content` и в непотоке кладёт ответ в `message.content`, а не в `response`:

```json
{
  "model": "qwen3:8b",
  "message": {"role": "assistant", "content": "ответ"},
  "done": true,
  "done_reason": "stop",
  "effective_ctx": 4096,
  "clamped": false,
  "context_length": 4096
}
```

Поток, обе ручки, NDJSON:

```json
{"model":"qwen3:8b","response":"ответ","done":false}
{"model":"qwen3:8b","response":"","done":true,"done_reason":"stop","effective_ctx":4096,"clamped":false}
```

`think`:

- поле отсутствует или не boolean — текст не меняется, блок `<think>` остаётся;
- `false` — вырезается `<think>...</think>`, в том числе незакрытый хвост `<think>` или `</think>`;
- `true` — блок остаётся.

Имена в `GET /api/tags`: имя файла GGUF основной модели, алиас `fastagg` и алиас `qwen3:8b`, чтобы текущий дефолт RAG отвечал без смены `OLLAMA_MODEL`.

`GET /api/ps`, пока кластер в фазе running:

```json
{"models": [{"name": "model.gguf", "context_length": 4096, "effective_ctx": 4096}]}
```

Пустой список, если модель не запущена.

`POST /api/show` с телом `{"name":"qwen3:8b"}`:

```json
{
  "model": "qwen3:8b",
  "parameters": "num_ctx 4096",
  "context_length": 4096,
  "effective_ctx": 4096,
  "details": {"family": "fastagg", "parameter_count": ""}
}
```

`context_length` и `effective_ctx` — контекст запущенного llama-server, не запрошенный `num_ctx`.

`POST /api/pull` принимает ссылку Hugging Face или имя с `/` и отвечает, что качать нужно с вкладки «Модели» или `fastagg pull`. Имя реестра Ollama вроде `qwen3:8b` — `400`. `POST /api/create` всегда `400`: Modelfile здесь не собирается.

`GET /api/version` сохраняет форму Ollama (одно строковое поле `version` — выпуск, например `0.2.0`) и добавляет `fastagg_version`. В сборке CI туда дописан номер прогона: `{"version":"0.2.0","fastagg_version":"0.2.0+14"}`. Те же два поля есть в `GET /fastagg/health` и `GET /api/health`.

## 3. Очередь

Одна генерация за раз (`limit = 1`) на дашборд-чат, `/api/generate`, `/api/chat` и OpenAI `/v1/chat/completions` с `/v1/completions` совместимого порта. Ждущих мест по умолчанию 8. Девятый одновременный запрос, когда одно место занято и 8 уже ждут, получает сразу, без долгого ожидания:

`429`, заголовок `Retry-After: 2`, тело `{"error":"Очередь генерации заполнена"}`.

Фоновый замер скорости уступает этой очереди. Эмбеддинги и реранкер в неё не входят.

Модель кластера не запущена: `503` и `{"error":"Основная модель кластера не запущена"}`.

## 4. Падение узла

`GET /fastagg/health` и `GET /api/health` отвечают одинаково:

```json
{
  "ok": true,
  "roles": {
    "llm": {"available": true, "model": "model-Q4_K_M.gguf", "effective_ctx": 4096, "replacing": false},
    "embedding": {
      "available": false,
      "replacing": true,
      "status": "replacing",
      "node_id": "node-b",
      "port": 18081,
      "embedding_model_id": "bge-m3-Q8_0",
      "gguf": "bge-m3-Q8_0.gguf",
      "quant": "Q8_0",
      "dim": 1024,
      "detail": "попытка 1 из 3: порт не открылся"
    },
    "reranker": {"available": false, "replacing": false, "detail": "роль не назначена"}
  },
  "failover_budget_sec": 90,
  "recommendation": "При 503 RAG может временно вернуться к локальному Ollama (профиль compose ollama).",
  "queue": {"active": 0, "waiting": 0, "limit": 1, "max_waiting": 8}
}
```

`available: true` только при статусе `up`. `replacing: true` — перенос идёт. После третьей неудачи статус `down`, `replacing: false`, эмбеддинги и реранкер отвечают `503`.

Как быстро это происходит. Опрос узлов каждые 3 с. Узел считается выбывшим через 12 с без успешного ответа (TTL реестра). Затем координатор до 3 раз запускает llama-server роли на другом узле со свободной памятью, по возможности вне основного разреза. Каждый запуск ждёт порт до 25 с. Вместе с ожиданием TTL бюджет около 90 с (`failover_budget_sec`). Память роли (около 1400 МиБ эмбеддинги, 1200 МиБ реранкер) вычитается из бюджета этого узла для основного разреза.

Пока роль `assigned` и на узле нет `model_path` к GGUF, `available` остаётся false: процесс не притворяется живым.

Рекомендация для RAG: держать локальный Ollama выключенным в обычной работе и включать его одним профилем, если health роли `llm` долго `available: false`. Образ и том `ollamadata` override не удаляет.

```text
docker compose -f docker-compose.yml -f integrations/rag5060/docker-compose.fastagg.yml --profile core --profile ollama up -d
```

На время локального Ollama остановите кластерную модель fastagg, иначе обе займут одну RTX 5060 Ti.

## Пример окружения

Файл `integrations/rag5060/.env.example`:

```env
OLLAMA_HOST=http://host.docker.internal:11435
OLLAMA_MODEL=qwen3:8b
EMBEDDING_URL=http://host.docker.internal:11435/v1/embeddings
RERANK_URL=http://host.docker.internal:11435/v1/rerank
FASTAGG_TOKEN=
```

С хоста Windows, без Docker, те же URL с `127.0.0.1` вместо `host.docker.internal`.
`EMBEDDING_URL` и `RERANK_URL` текущий RAG-5060 ещё не читает: их добавит разработчик RAG. Контракт выше для этого изменения.
