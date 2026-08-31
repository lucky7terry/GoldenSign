/**
 * WebView 진입점.
 *
 * WebView 에는 네이티브 접근 권한이 전혀 없다. 하드웨어 관련 동작은 전부
 * `mentra.send(channel, payload)` 로 background 에 넘기고, background 가
 * `session.*` SDK 호출로 번역한다.
 */

import {createRoot} from "react-dom/client"
import {MentraProvider} from "@mentra/miniapp/ui"
import "../shared/channels"
import {App} from "./App"
import "./styles.css"

const root = document.getElementById("root")
if (!root) {
  throw new Error("Missing #root element in index.html")
}

// MentraProvider 는 호스트 컬러 스킴에 맞춰 <html class="dark"> 를 토글할 뿐이다.
// CSS 테마 브리지이며 MiniappSession 을 만들지 않는다.
createRoot(root).render(
  <MentraProvider>
    <App />
  </MentraProvider>,
)

// mentra.ready() 를 여기서 부르지 않는 이유: render() 는 concurrent 라서 이 줄에서
// 호출하면 App 의 구독이 걸리기 전에 도달한다. App 의 mount effect 안, 채널 핸들러
// 등록 직후로 옮겼다. SDK 가 멱등이라고 명시하므로 호출 지점만 옮긴 것이다.
