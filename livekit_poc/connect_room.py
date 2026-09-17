import asyncio
import os

from dotenv import load_dotenv
from livekit import api, rtc


load_dotenv(".env.local", override=True)


ROOM_NAME = "agni-ai-voice-poc"
PARTICIPANT_ID = "diya-poc"


def create_token() -> str:
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY or LIVEKIT_API_SECRET is missing from .env.local"
        )

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(PARTICIPANT_ID)
        .with_name("Agni AI POC")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    return token


async def main():
    livekit_url = os.getenv("LIVEKIT_URL")

    if not livekit_url:
        raise RuntimeError("LIVEKIT_URL is missing from .env.local")

    token = create_token()

    room = rtc.Room()

    print("Connecting to LiveKit...")

    await room.connect(
        livekit_url,
        token,
    )

    print("CONNECTED TO LIVEKIT")
    print("Room:", room.name)
    print("Participant:", room.local_participant.identity)

    await asyncio.sleep(5)

    await room.disconnect()

    print("DISCONNECTED FROM LIVEKIT")


if __name__ == "__main__":
    asyncio.run(main())