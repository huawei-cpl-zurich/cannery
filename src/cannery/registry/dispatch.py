import dramatiq
from dramatiq.brokers.redis import RedisBroker


def publish_registration(redis_url: str, key: str) -> None:
    broker = RedisBroker(  # type: ignore[no-untyped-call]
        url=redis_url,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    try:
        broker.enqueue(
            dramatiq.Message(
                queue_name="registration",
                actor_name="register_spec",
                args=(key,),
                kwargs={},
                options={},
            )
        )
    finally:
        broker.close()
