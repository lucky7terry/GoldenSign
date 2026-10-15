# 배포 가이드

AI 서버를 외부에서 접속할 수 있는 HTTPS 주소로 띄운다.
구성은 `docker-compose.yml` 하나다 — AI 서버 컨테이너와, 그 앞에서 HTTPS 인증서를
자동으로 받아 주는 Caddy 컨테이너.

```
미니앱 ──https/wss──▶ Caddy(443) ──▶ server(8000)
                                     │
Mentra 스트림(Cloudflare) ◀──WebRTC──┘  서버가 영상을 당겨 온다
```

## 서버 사양

| 항목 | 값 | 근거 |
|---|---|---|
| 메모리 | 4GB | 실측 최대 약 850MB(1080p 처리) + OS·Docker·설치 여유 |
| CPU | 2 vCPU | 초당 12~15장 처리. 10장 이상이면 정확도 변화 없음 |
| OS | Ubuntu 24.04 | Python 은 컨테이너 안에서 3.12 로 고정 |
| 열 포트 | 22, 80, 443 | 80 은 인증서 발급에 쓴다. WebRTC 는 서버가 먼저 나가므로 따로 열 필요 없음 |

Lightsail 일반 요금제는 CPU 기준치가 20% 인 버스트 방식이다. 스트리밍을 몇 시간씩
계속 켜 두면 크레딧이 떨어져 처리 속도가 크게 낮아질 수 있다. 콘솔의
**Metrics → CPU burst capacity** 로 남은 양을 볼 수 있다.

## 1. 무료 주소 만들기 (DuckDNS)

1. https://www.duckdns.org 에 GitHub/Google 계정으로 로그인
2. 원하는 이름으로 서브도메인 추가 (예: `goldensign` → `goldensign.duckdns.org`)
3. 서버를 만든 뒤 **고정 IP** 를 `current ip` 칸에 넣고 `update ip`

고정 IP 를 쓰므로 IP 자동 갱신 스크립트는 필요 없다.

## 2. 서버 만들기 (Lightsail)

1. 리전 **서울(ap-northeast-2)**, 플랫폼 Linux, 블루프린트 **OS Only → Ubuntu 24.04**
2. 요금제 선택 후 생성
3. **Networking → Create static IP** 로 고정 IP 를 만들어 인스턴스에 붙인다
   (붙어 있는 동안은 무료, 떼어 둔 채로 두면 과금)
4. 같은 화면 **IPv4 Firewall** 에 HTTPS(443) 를 추가한다 (22, 80 은 기본으로 열려 있음)
5. 1번의 DuckDNS 에 이 고정 IP 를 넣는다

## 3. 서버 준비

브라우저 SSH(인스턴스 화면의 `Connect using SSH`)로 접속해서:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit   # 다시 접속해야 docker 그룹이 적용된다
```

```bash
git clone https://github.com/lucky7terry/GoldenSign.git
cd GoldenSign/infra
mkdir -p models
cp .env.example .env
nano .env   # DOMAIN 을 1번에서 만든 주소로
```

저장소가 비공개라면 clone 할 때 GitHub 개인 액세스 토큰이 필요하다.

## 4. 모델 파일 올리기

인식 모델(.keras)은 저장소에 없다. 내 컴퓨터에서(PowerShell):

```powershell
scp -i LightsailDefaultKey.pem server\models\sign_transformer_test_yewon.keras ubuntu@<고정IP>:~/GoldenSign/infra/models/
```

키 파일(`.pem`)은 Lightsail **Account → SSH keys** 에서 받는다.
MediaPipe `.task` 파일은 컨테이너가 처음 뜰 때 자동으로 받는다.

## 5. 실행

```bash
cd ~/GoldenSign/infra
docker compose up -d --build
```

첫 빌드는 TensorFlow 설치 때문에 몇 분 걸린다.

## 6. 확인

```bash
curl https://<DOMAIN>/health
```

`"loaded":true`, `"mode":"recognition"` 이 나오면 된다.

```bash
docker compose logs -f server
```

미니앱으로 스트림을 시작했을 때 이 줄들이 보여야 한다.

| 로그 | 의미 |
|---|---|
| `WHEP connection established` | 안경 영상 WebRTC 연결 성공 |
| `Processed WHEP video frames ... fps=` | 처리 속도. 10 이상이면 정상 |

## 7. 미니앱 연결

`miniapp/src/shared/config.ts` 의 `AI_HTTP` 를 `https://<DOMAIN>` 으로 바꾼다.
`ws_url` 은 서버가 `wss://<DOMAIN>/...` 로 만들어 준다.

## 운영

| 할 일 | 명령 |
|---|---|
| 코드 갱신 | `git pull && docker compose up -d --build` |
| 모델 교체 | `infra/models/` 에 새 파일을 넣고 `.env` 의 파일명 수정 → `docker compose up -d` |
| 재시작 | `docker compose restart server` |
| 중지 | `docker compose down` |

## 종료 (프로젝트가 끝나면)

인스턴스는 **정지해도 요금이 나간다.** Lightsail 콘솔에서

1. 인스턴스 **Delete**
2. 고정 IP **Release**
3. 스냅샷을 만들었다면 함께 삭제
