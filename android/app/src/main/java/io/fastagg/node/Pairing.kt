package io.fastagg.node

import android.content.Context
import java.io.OutputStreamWriter
import java.net.URL
import java.security.MessageDigest
import java.security.cert.X509Certificate
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

fun pairWithCoordinator(context: Context, join: JoinLink): String {
    val url = URL("https://${join.host}:${join.port}/api/pair")
    val connection = url.openConnection() as HttpsURLConnection
    connection.sslSocketFactory = pinnedContext(join.fingerprint).socketFactory
    connection.hostnameVerifier = HostnameVerifier { _, _ -> true }
    connection.requestMethod = "POST"
    connection.doOutput = true
    connection.connectTimeout = 8000
    connection.readTimeout = 8000
    connection.setRequestProperty("Content-Type", "application/json")
    val prefs = context.getSharedPreferences("fastagg", Context.MODE_PRIVATE)
    val nodeId = prefs.getString("node_id", null) ?: java.util.UUID.randomUUID().toString().also {
        prefs.edit().putString("node_id", it).apply()
    }
    val body = """{"code":${jsonString(join.code)},"node_id":${jsonString(nodeId)},"name":${jsonString(android.os.Build.MODEL)},"advertise_host":${jsonString(localIpv4())},"http_port":47000,"rpc_port":50052}"""
    OutputStreamWriter(connection.outputStream, Charsets.UTF_8).use { it.write(body) }
    val code = connection.responseCode
    val text = (if (code in 200..299) connection.inputStream else connection.errorStream)
        ?.bufferedReader()?.use { it.readText() } ?: ""
    if (code !in 200..299) throw IllegalStateException("Координатор ответил $code: $text")
    val token = Regex(""""token"\s*:\s*"([^"]+)"""").find(text)?.groupValues?.get(1)
        ?: throw IllegalStateException("Координатор не выдал токен")
    return token
}

fun saveMembership(context: Context, join: JoinLink, token: String) {
    context.getSharedPreferences("fastagg", Context.MODE_PRIVATE).edit()
        .putString("token", token)
        .putString("host", join.host)
        .putInt("port", join.port)
        .putString("fingerprint", join.fingerprint)
        .apply()
}

private fun pinnedContext(expected: String): SSLContext {
    val trust = object : X509TrustManager {
        override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) = Unit
        override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
        override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {
            val cert = chain?.firstOrNull() ?: throw IllegalStateException("Нет сертификата")
            val digest = MessageDigest.getInstance("SHA-256").digest(cert.encoded)
            val got = digest.joinToString(":") { "%02x".format(it) }
            if (!got.equals(expected.replace(" ", ""), ignoreCase = true)) {
                throw IllegalStateException("Отпечаток $got не совпал")
            }
        }
    }
    val context = SSLContext.getInstance("TLS")
    context.init(null, arrayOf<TrustManager>(trust), null)
    return context
}

private fun jsonString(value: String): String = "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

fun localIpv4(): String {
    val interfaces = java.util.Collections.list(java.net.NetworkInterface.getNetworkInterfaces())
    for (item in interfaces) {
        val addresses = java.util.Collections.list(item.inetAddresses)
        for (address in addresses) {
            if (!address.isLoopbackAddress && address is java.net.Inet4Address) {
                return address.hostAddress ?: continue
            }
        }
    }
    return "127.0.0.1"
}
