package com.inverse

import android.annotation.SuppressLint
import android.app.Activity
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebView

class DeepLinkActivity : Activity() {
    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val webView = WebView(this)
        webView.settings.javaScriptEnabled = true
        webView.addJavascriptInterface(AccountBridge(), "Account")
        setContentView(webView)
        val target = intent.getStringExtra("url") ?: return
        webView.loadUrl(target)
    }
}

class AccountBridge {
    @JavascriptInterface
    fun accountId(): String = "current-account"
}
