package io.fastagg.node

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.zxing.integration.android.IntentIntegrator

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val link = findViewById<EditText>(R.id.link)
        val status = findViewById<TextView>(R.id.status)
        intent?.data?.toString()?.let { if (it.startsWith("fastagg://")) link.setText(it) }
        if (NodeService.running) status.text = "Узел уже работает"
        requestNeededPermissions()
        findViewById<Button>(R.id.scan).setOnClickListener {
            IntentIntegrator(this).setDesiredBarcodeFormats(IntentIntegrator.QR_CODE).setPrompt("Код fastagg").initiateScan()
        }
        findViewById<Button>(R.id.pair).setOnClickListener {
            val raw = link.text.toString()
            status.text = "Сопряжение…"
            Thread {
                val message = try {
                    val join = parseJoinLink(raw)
                    val token = pairWithCoordinator(this, join)
                    saveMembership(this, join, token)
                    val service = Intent(this, NodeService::class.java)
                    if (Build.VERSION.SDK_INT >= 26) startForegroundService(service) else startService(service)
                    "Узел запущен"
                } catch (error: Exception) {
                    error.message ?: "Не удалось сопрячься"
                }
                runOnUiThread { status.text = message }
            }.start()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val result = IntentIntegrator.parseActivityResult(requestCode, resultCode, data)
        if (result != null && result.contents != null) {
            findViewById<EditText>(R.id.link).setText(result.contents)
        } else {
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    private fun requestNeededPermissions() {
        val needed = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 33) needed += Manifest.permission.POST_NOTIFICATIONS
        val missing = needed.filter { ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) ActivityCompat.requestPermissions(this, missing.toTypedArray(), 1)
    }
}
