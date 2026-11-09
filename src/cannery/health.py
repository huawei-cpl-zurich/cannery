"""Dependency probes with a hard client-side deadline."""

import asyncio

from psycopg import AsyncConnection, Error


async def postgres_ready(url: str, timeout: float = 2) -> bool:
    connection = None
    query = None
    try:
        async with asyncio.timeout(timeout):
            connection = await AsyncConnection.connect(
                url.replace("postgresql+psycopg://", "postgresql://", 1),
                autocommit=True,
                connect_timeout=2,
            )
            query = asyncio.create_task(connection.execute("SELECT 1"))
            done, _ = await asyncio.wait({query}, timeout=timeout / 2)
            if not done:
                return False
            await query
            return True
    except (Error, TimeoutError):
        return False
    finally:
        if connection is not None:
            # Close the socket before cancelling: PostgreSQL cancellation itself
            # needs another network roundtrip and can stall during an outage.
            connection.pgconn.finish()
        if query is not None and not query.done():
            query.cancel()
            await asyncio.gather(query, return_exceptions=True)
        if connection is not None:
            await connection.close()
