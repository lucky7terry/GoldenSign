/**
 * Shared build-time configuration for the miniapp.
 *
 * Imported by the background JSContext. Values are inlined by the bundler —
 * there is no runtime config fetch and no env var access (the bare JS engine
 * has no `process`).
 */

/**
 * AI server base URL.
 *
 * MUST be the Mac's LAN IP, not localhost/127.0.0.1 — the background bundle
 * runs on the *phone*, so loopback would resolve to the phone itself.
 *
 * Re-check this whenever the Mac changes network (café Wi-Fi, hotspot, VPN):
 *   ipconfig getifaddr en0
 *
 * Cleartext http:// to a private IP is the thing Gate 3 is probing. iOS ATS
 * and Android's cleartext policy can each block it; ai-client.ts logs enough
 * to tell that apart from "server not running".
 */
export const AI_HTTP = "http://192.168.35.161:8000"

/**
 * Schema version for the session/handshake message family
 * (hello / ready / frame / result / ack / stop).
 * Matches server/app/constants.py SCHEMA_VERSION.
 */
export const HELLO_SCHEMA = "dev-0.2"

/**
 * Schema version for the WebRTC stream message family
 * (stream_start / stream_stop). Matches server/app/constants.py
 * WEBRTC_SCHEMA_VERSION. Unused in Gate 3 — no stream is started here.
 */
export const STREAM_SCHEMA = "dev-0.3"

/** Sent as `client` on POST /v1/sessions so server logs can attribute sessions. */
export const CLIENT_NAME = "mentra-local-miniapp"
