package io.fastagg.node

import org.junit.Assert.assertEquals
import org.junit.Test

class ProtocolTest {
    @Test
    fun joinLinkAndMacMatchPython() {
        val link = parseJoinLink("fastagg://192.168.1.20:8080?code=ABCD2345&fp=aa:bb")
        assertEquals("192.168.1.20", link.host)
        assertEquals(8080, link.port)
        assertEquals("ABCD2345", link.code)
        assertEquals("aa:bb", link.fingerprint)
        val fields = linkedMapOf<String, Any>(
            "advertise_host" to "192.168.1.20",
            "arch" to "arm64",
            "http_port" to 47000,
            "kind" to "announce",
            "magic" to "FAAG1",
            "name" to "phone",
            "node_id" to "android-test",
            "os_name" to "Android",
            "preferred_backend" to "cpu",
            "rpc_port" to 50052,
            "software" to "fast-aggregation",
            "v" to 1,
        )
        assertEquals(
            """{"advertise_host":"192.168.1.20","arch":"arm64","http_port":47000,"kind":"announce","magic":"FAAG1","name":"phone","node_id":"android-test","os_name":"Android","preferred_backend":"cpu","rpc_port":50052,"software":"fast-aggregation","v":1}""",
            canonicalAnnounce(fields),
        )
        assertEquals(
            "948127acaf412c5f53a7dd944e3e5ab1c71f35a55ddb5328f2978b5636fc7820",
            announceMac(fields, "secret"),
        )
    }
}
