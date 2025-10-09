import asyncio
from services.spotify.client import SpotifyClient

async def main():
    client = SpotifyClient()
    tracks = await client.search_tracks('The Weeknd', limit=1)
    print(tracks)

asyncio.run(main())
