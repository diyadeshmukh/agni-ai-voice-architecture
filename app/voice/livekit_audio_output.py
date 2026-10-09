"""
Jeeva AI - LiveKit Audio Output

Publishes synthesized speech audio into a LiveKit room.

This module is intentionally independent of any particular
TTS provider.

Flow:

    TTSProvider
        ↓
    TTSAudioChunk
        ↓
    PCM audio
        ↓
    LiveKit AudioSource
        ↓
    LocalAudioTrack
        ↓
    LiveKit Room
        ↓
    Caller / Remote Participant
"""

from __future__ import annotations

import asyncio

from livekit import rtc

from app.voice.tts_provider import TTSAudioChunk


class LiveKitAudioOutput:
    """
    Handles publishing synthesized speech audio through LiveKit.

    Supports hard interruption so an active AI response can be
    stopped immediately when the user barges in.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
    ) -> None:
        """
        Initialize the LiveKit audio output.

        Args:
            sample_rate:
                PCM sample rate used by the TTS provider.

            channels:
                Number of PCM audio channels.
        """

        self.sample_rate = sample_rate
        self.channels = channels

        # -----------------------------------------------------------
        # LiveKit audio source
        # -----------------------------------------------------------

        self.audio_source = rtc.AudioSource(
            sample_rate=self.sample_rate,
            num_channels=self.channels,
        )

        # -----------------------------------------------------------
        # Outgoing LiveKit track
        # -----------------------------------------------------------

        self.audio_track = (
            rtc.LocalAudioTrack.create_audio_track(
                "voice-output",
                self.audio_source,
            )
        )

        # -----------------------------------------------------------
        # Native capture operations
        # -----------------------------------------------------------
        #
        # capture_frame() crosses into LiveKit's native layer.
        #
        # Keep references to active operations so shutdown can wait
        # for them safely instead of abandoning them mid-operation.

        self._capture_tasks: set[
            asyncio.Task[None]
        ] = set()

        # -----------------------------------------------------------
        # Hard interruption state
        # -----------------------------------------------------------
        #
        # Once a user interruption is confirmed, remaining audio
        # chunks from the current AI response must not be played.
        #
        # begin_response() resets this for the next AI turn.

        self._hard_interrupted = False

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(
        self,
        participant: rtc.LocalParticipant,
    ) -> rtc.LocalTrackPublication:
        """
        Publish the AI voice track into the LiveKit room.
        """

        return await participant.publish_track(
            self.audio_track
        )

    # ------------------------------------------------------------------
    # New response
    # ------------------------------------------------------------------

    def begin_response(self) -> None:
        """
        Prepare the audio output for a new AI response.

        A previous user barge-in may have left the output in the
        interrupted state. The next response starts fresh.
        """

        self._hard_interrupted = False

    # ------------------------------------------------------------------
    # Send TTS chunk
    # ------------------------------------------------------------------

    async def send_chunk(
        self,
        audio_chunk: TTSAudioChunk,
    ) -> None:
        """
        Send one synthesized PCM audio chunk into LiveKit.

        If the current AI response has already been interrupted,
        the chunk is discarded.
        """

        # -----------------------------------------------------------
        # Validate channel count
        # -----------------------------------------------------------

        if audio_chunk.channels != self.channels:

            raise ValueError(
                "TTS audio channel count does not match "
                "LiveKit audio output configuration"
            )

        # -----------------------------------------------------------
        # Validate encoding
        # -----------------------------------------------------------

        if audio_chunk.encoding != "pcm_s16le":

            raise ValueError(
                "LiveKit audio output currently expects "
                "PCM 16-bit little-endian audio"
            )

        # -----------------------------------------------------------
        # Drop audio belonging to an interrupted response
        # -----------------------------------------------------------

        if self._hard_interrupted:
            return

        # -----------------------------------------------------------
        # Build LiveKit audio frame
        # -----------------------------------------------------------

        frame = rtc.AudioFrame(
            data=audio_chunk.data,
            sample_rate=audio_chunk.sample_rate,
            num_channels=audio_chunk.channels,
            samples_per_channel=(
                len(audio_chunk.data)
                // 2
                // audio_chunk.channels
            ),
        )

        # -----------------------------------------------------------
        # Send frame into LiveKit
        # -----------------------------------------------------------

        capture_task = asyncio.create_task(
            self.audio_source.capture_frame(
                frame
            )
        )

        self._capture_tasks.add(
            capture_task
        )

        try:

            # Shield the native LiveKit operation from abrupt Python
            # cancellation.
            #
            # This avoids leaving capture_frame() half-finished while
            # still allowing us to track it for clean shutdown.

            await asyncio.shield(
                capture_task
            )

        finally:

            if capture_task.done():

                self._capture_tasks.discard(
                    capture_task
                )

            else:

                capture_task.add_done_callback(
                    self._capture_tasks.discard
                )

        # -----------------------------------------------------------
        # Handle interruption race
        # -----------------------------------------------------------
        #
        # The user may have interrupted Jeeva while capture_frame()
        # was completing.
        #
        # In that case the frame may have entered LiveKit's queue
        # just before interrupt() cleared it.
        #
        # Clear the queue once more to guarantee that old-response
        # audio does not continue playing.

        if self._hard_interrupted:

            self.audio_source.clear_queue()

    # ------------------------------------------------------------------
    # Playback completion
    # ------------------------------------------------------------------

    async def wait_for_playout(self) -> None:
        """
        Wait until all queued AI audio has finished playing.
        """

        await self.audio_source.wait_for_playout()

    # ------------------------------------------------------------------
    # Hard barge-in
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """
        Immediately stop the current AI voice response.

        This performs the audio side of hard barge-in:

            mark current response interrupted
                    ↓
            discard queued LiveKit audio
                    ↓
            reject remaining TTS chunks

        The LLM/TTS worker is cancelled separately by the voice
        pipeline using interrupt_event.
        """

        self._hard_interrupted = True

        # Immediately discard any speech already queued inside
        # LiveKit's AudioSource.

        self.audio_source.clear_queue()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Cleanly release the LiveKit audio source.
        """

        # Prevent additional chunks from the active response from
        # being accepted during shutdown.

        self._hard_interrupted = True

        # -----------------------------------------------------------
        # Finish native capture operations
        # -----------------------------------------------------------

        if self._capture_tasks:

            await asyncio.gather(
                *tuple(
                    self._capture_tasks
                ),
                return_exceptions=True,
            )

        # -----------------------------------------------------------
        # Remove any remaining queued speech
        # -----------------------------------------------------------

        self.audio_source.clear_queue()

        # -----------------------------------------------------------
        # Close LiveKit AudioSource
        # -----------------------------------------------------------

        await self.audio_source.aclose()