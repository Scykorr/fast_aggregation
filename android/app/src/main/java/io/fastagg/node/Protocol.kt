package io.fastagg.node

import java.net.URLDecoder
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

data class JoinLink(val host: String, val port: Int, val code: String, val fingerprint: String)

fun parseJoinLink(raw: String): JoinLink {
    val text = raw.trim()
    if (!text.startsWith("fastagg://")) {
        throw IllegalArgumentException("Ожидалась ссылка fastagg://host:port?code=...&fp=...")
    }
    val withoutScheme = text.removePrefix("fastagg://")
    val hostPart = withoutScheme.substringBefore("?")
    val query = withoutScheme.substringAfter("?", "")
    val host = hostPart.substringBefore(":")
    val port = hostPart.substringAfter(":", "").toIntOrNull() ?: 0
    val params = query.split("&").mapNotNull { item ->
        val key = item.substringBefore("=")
        val value = URLDecoder.decode(item.substringAfter("=", ""), "UTF-8")
        if (key.isEmpty()) null else key to value
    }.toMap()
    val code = params["code"] ?: ""
    val fingerprint = params["fp"] ?: ""
    if (host.isEmpty() || port !in 1..65535 || code.isEmpty() || fingerprint.isEmpty()) {
        throw IllegalArgumentException("В ссылке нужны адрес, код и отпечаток")
    }
    return JoinLink(host, port, code, fingerprint)
}

fun canonicalAnnounce(fields: Map<String, Any>): String {
    val body = fields.toSortedMap().entries.joinToString(",") { (key, value) ->
        "\"$key\":${jsonLiteral(value)}"
    }
    return "{$body}"
}

fun announceMac(fields: Map<String, Any>, token: String): String {
    val mac = Mac.getInstance("HmacSHA256")
    mac.init(SecretKeySpec(token.toByteArray(Charsets.UTF_8), "HmacSHA256"))
    return mac.doFinal(canonicalAnnounce(fields).toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }
}

private fun jsonLiteral(value: Any): String = when (value) {
    is Number -> value.toString()
    else -> "\"" + value.toString().replace("\\", "\\\\").replace("\"", "\\\"") + "\""
}
