package io.fastagg.node

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

class NodeService : Service() {
    private val stop = AtomicBoolean(false)
    private var http: ServerSocket? = null
    private var rpc: Process? = null
    private var rpcBackend: String? = null
    private var statusText = "запуск"

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        running = true
        createChannel()
        val notice = notification("узел запускается")
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(1, notice, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(1, notice)
        }
        stop.set(false)
        Thread { httpLoop() }.start()
        Thread { announceLoop() }.start()
        Thread { guardLoop() }.start()
        return START_STICKY
    }

    override fun onDestroy() {
        stop.set(true)
        running = false
        stopRpc()
        http?.close()
        super.onDestroy()
    }

    private fun httpLoop() {
        val server = ServerSocket(47000)
        http = server
        statusText = "слушает 47000"
        update("слушает порт 47000")
        while (!stop.get()) {
            val client = try {
                server.accept()
            } catch (_: Exception) {
                break
            }
            Thread { handle(client) }.start()
        }
    }

    private fun handle(socket: Socket) {
        socket.use { client ->
            client.soTimeout = 5000
            val reader = BufferedReader(InputStreamReader(client.getInputStream()))
            val request = reader.readLine() ?: return
            val headers = mutableMapOf<String, String>()
            while (true) {
                val line = reader.readLine() ?: break
                if (line.isEmpty()) break
                val index = line.indexOf(':')
                if (index > 0) headers[line.substring(0, index).lowercase()] = line.substring(index + 1).trim()
            }
            val path = request.split(" ").getOrNull(1)?.substringBefore("?") ?: "/"
            val method = request.substringBefore(" ")
            if (path != "/health" && !authorized(headers["authorization"])) {
                write(client, 401, """{"detail":"Нужен токен кластера"}""", "application/json")
                return
            }
            when {
                method == "GET" && path == "/health" ->
                    write(client, 200, """{"status":"ok","software":"fast-aggregation"}""", "application/json")
                method == "GET" && path == "/v1/node" ->
                    write(client, 200, nodeJson(), "application/json")
                method == "GET" && path == "/v1/bench" ->
                    write(client, 200, "x".repeat(1024), "application/octet-stream")
                method == "POST" && path == "/v1/rpc/start" -> {
                    startRpc()
                    write(client, 200, """{"running":${rpc?.isAlive == true},"backend":${rpcBackend?.let { '"' + it + '"' } ?: "null"},"error":null}""", "application/json")
                }
                method == "POST" && path == "/v1/rpc/stop" -> {
                    stopRpc()
                    write(client, 200, """{"running":false}""", "application/json")
                }
                else -> write(client, 404, """{"detail":"нет"}""", "application/json")
            }
        }
    }

    private fun authorized(header: String?): Boolean {
        val token = prefs().getString("token", null) ?: return true
        return header == "Bearer $token"
    }

    private fun startRpc() {
        if (rpc?.isAlive == true) return
        val dir = applicationInfo.nativeLibraryDir
        val opencl = File(dir, "libfastagg-rpc-opencl.so")
        val cpu = File(dir, "libfastagg-rpc-cpu.so")
        val binary = if (hasAdreno() && opencl.exists()) opencl else cpu
        if (!binary.exists()) {
            statusText = "нет вшитого ggml-rpc-server"
            update(statusText)
            return
        }
        rpcBackend = if (binary == opencl) "opencl" else "cpu"
        val process = ProcessBuilder(binary.absolutePath, "--host", "0.0.0.0", "--port", "50052")
            .directory(File(dir))
            .redirectErrorStream(true)
            .apply { environment()["LD_LIBRARY_PATH"] = dir }
            .start()
        rpc = process
        Thread.sleep(400)
        if (!process.isAlive && binary == opencl && cpu.exists()) {
            rpcBackend = "cpu"
            rpc = ProcessBuilder(cpu.absolutePath, "--host", "0.0.0.0", "--port", "50052")
                .directory(File(dir))
                .redirectErrorStream(true)
                .apply { environment()["LD_LIBRARY_PATH"] = dir }
                .start()
        }
        statusText = "RPC ${rpcBackend}"
        update(statusText)
    }

    private fun stopRpc() {
        rpc?.destroy()
        rpc = null
        rpcBackend = null
        statusText = "RPC остановлен"
        update(statusText)
    }

    private fun guardLoop() {
        while (!stop.get()) {
            val temp = hottestCelsius()
            val battery = batteryPercent()
            val plugged = isPlugged()
            if (rpc?.isAlive == true && (temp != null && temp >= 92.0 || battery != null && battery < 35 && !plugged)) {
                stopRpc()
                statusText = if (temp != null && temp >= 92.0) "стоп: перегрев" else "стоп: низкий заряд"
                update(statusText)
            }
            Thread.sleep(15000)
        }
    }

    private fun announceLoop() {
        val socket = DatagramSocket()
        socket.broadcast = true
        while (!stop.get()) {
            try {
                val packet = announcePacket()
                val data = packet.toByteArray()
                socket.send(DatagramPacket(data, data.size, InetAddress.getByName("255.255.255.255"), 45221))
            } catch (_: Exception) {
            }
            Thread.sleep(3000)
        }
        socket.close()
    }

    private fun announcePacket(): String {
        val fields = linkedMapOf<String, Any>(
            "advertise_host" to localIpv4(),
            "arch" to "arm64",
            "http_port" to 47000,
            "kind" to "announce",
            "magic" to "FAAG1",
            "name" to (android.os.Build.MODEL ?: "android"),
            "node_id" to nodeId(),
            "os_name" to "Android",
            "preferred_backend" to if (hasAdreno()) "opencl" else "cpu",
            "rpc_port" to 50052,
            "software" to "fast-aggregation",
            "v" to 1,
            "version" to BuildConfig.VERSION_NAME,
        )
        val token = prefs().getString("token", null)
        val base = canonicalAnnounce(fields)
        if (token.isNullOrEmpty()) return base
        return base.dropLast(1) + ",\"mac\":\"" + announceMac(fields, token) + "\"}"
    }

    private fun nodeJson(): String {
        val memory = android.app.ActivityManager.MemoryInfo()
        (getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager).getMemoryInfo(memory)
        val total = memory.totalMem / (1024 * 1024)
        val free = memory.availMem / (1024 * 1024)
        val temp = hottestCelsius()
        val battery = batteryPercent()
        val plugged = isPlugged()
        val opencl = hasAdreno()
        return """{"node_id":"${nodeId()}","name":"${android.os.Build.MODEL}","software":"fast-aggregation","version":"${BuildConfig.VERSION_NAME}","advertise_host":"${localIpv4()}","http_port":47000,"rpc_port":50052,"preferred_backend":"${if (opencl) "opencl" else "cpu"}","resources":{"os_name":"Android","os_version":"${android.os.Build.VERSION.RELEASE}","arch":"arm64","cpu_model":"${android.os.Build.HARDWARE}","cpu_physical":null,"cpu_logical":${Runtime.getRuntime().availableProcessors()},"ram_total_mb":$total,"ram_available_mb":$free,"gpus":[],"backends":[{"name":"cpu","available":true,"detail":"вшитый ggml-rpc-server"},{"name":"opencl","available":$opencl,"detail":"Adreno kgsl"}],"cpu_percent":null,"temperature_c":${temp ?: "null"},"battery_percent":${battery ?: "null"},"power_plugged":$plugged,"device_class":"phone"},"rpc":{"running":${rpc?.isAlive == true},"backend":${rpcBackend?.let { "\"$it\"" } ?: "null"},"error":null,"log_tail":[]}}"""
    }

    private fun hottestCelsius(): Double? {
        val root = File("/sys/class/thermal")
        val zones = root.listFiles { file -> file.isDirectory && file.name.startsWith("thermal_zone") } ?: return null
        return zones.mapNotNull { zone ->
            val raw = File(zone, "temp").takeIf { it.isFile }?.readText()?.trim()?.toDoubleOrNull()
            raw?.let { if (it > 200) it / 1000.0 else it }
        }.filter { it in 1.0..120.0 }.maxOrNull()
    }

    private fun batteryPercent(): Int? {
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return null
        val level = battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = battery.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        if (level < 0 || scale <= 0) return null
        return level * 100 / scale
    }

    private fun isPlugged(): Boolean {
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return true
        return battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
    }

    private fun hasAdreno(): Boolean = File("/sys/class/kgsl/kgsl-3d0").exists()

    private fun nodeId(): String {
        val prefs = prefs()
        return prefs.getString("node_id", null) ?: UUID.randomUUID().toString().also {
            prefs.edit().putString("node_id", it).apply()
        }
    }

    private fun prefs() = getSharedPreferences("fastagg", Context.MODE_PRIVATE)

    private fun update(text: String) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(1, notification(text))
    }

    private fun notification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("fastagg узел")
            .setContentText(text)
            .setOngoing(true)
            .build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < 26) return
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "fastagg", NotificationManager.IMPORTANCE_LOW))
    }

    private fun write(socket: Socket, code: Int, body: String, type: String) {
        val bytes = body.toByteArray()
        val reason = if (code == 200) "OK" else "Error"
        val header = "HTTP/1.1 $code $reason\r\nContent-Type: $type\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n\r\n"
        val out = socket.getOutputStream()
        out.write(header.toByteArray())
        out.write(bytes)
        out.flush()
    }

    companion object {
        @Volatile var running: Boolean = false
        private const val CHANNEL = "fastagg"
    }
}
