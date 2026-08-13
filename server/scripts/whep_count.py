import argparse
import asyncio
from urllib.parse import urljoin


async def wait_for_ice_gathering(peer_connection, timeout_seconds: float) -> None:
    if peer_connection.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @peer_connection.on("icegatheringstatechange")
    def on_ice_gathering_state_change():
        if peer_connection.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print("ICE gathering timed out; continuing with local description")


async def count_whep_frames(whep_url: str, timeout_seconds: float) -> None:
    import aiohttp
    from aiortc import RTCPeerConnection, RTCSessionDescription

    peer_connection = RTCPeerConnection()
    session = aiohttp.ClientSession()
    resource_url = None

    try:
        track_queue = asyncio.Queue(maxsize=1)

        @peer_connection.on("track")
        def on_track(track):
            if track.kind == "video" and track_queue.empty():
                track_queue.put_nowait(track)

        peer_connection.addTransceiver("video", direction="recvonly")
        offer = await peer_connection.createOffer()
        await peer_connection.setLocalDescription(offer)
        await wait_for_ice_gathering(peer_connection, timeout_seconds)

        async with session.post(
            whep_url,
            data=peer_connection.localDescription.sdp,
            headers={
                "Content-Type": "application/sdp",
                "Accept": "application/sdp",
            },
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as response:
            if response.status not in {200, 201}:
                body = await response.text()
                raise RuntimeError(
                    f"WHEP POST failed with status {response.status}: {body[:200]}"
                )
            answer_sdp = await response.text()
            location = response.headers.get("Location")
            if location:
                resource_url = urljoin(whep_url, location)

        await peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type="answer")
        )
        print("WHEP connected. Waiting for video frames...")

        video_track = await asyncio.wait_for(track_queue.get(), timeout=timeout_seconds)
        frame_count = 0
        while True:
            await video_track.recv()
            frame_count += 1
            if frame_count == 1 or frame_count % 30 == 0:
                print(f"received frames: {frame_count}")
    finally:
        if resource_url:
            try:
                async with session.delete(
                    resource_url,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ):
                    pass
            except Exception as exc:
                print(f"WHEP DELETE failed: {exc}")
        await session.close()
        await peer_connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Count frames from a WHEP URL.")
    parser.add_argument("whep_url")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    asyncio.run(count_whep_frames(args.whep_url, args.timeout))


if __name__ == "__main__":
    main()
