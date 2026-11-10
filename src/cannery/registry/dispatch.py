import dramatiq
from dramatiq.brokers.redis import RedisBroker


def publish(redis_url: str, key: str, queue: str, actor: str) -> None:
    broker = RedisBroker(  # type: ignore[no-untyped-call]
        url=redis_url,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    try:
        broker.enqueue(
            dramatiq.Message(
                queue_name=queue,
                actor_name=actor,
                args=(key,),
                kwargs={},
                options={},
            )
        )
    finally:
        broker.close()


def publish_registration(redis_url: str, key: str) -> None:
    publish(redis_url, key, "registration", "register_spec")


def publish_submission(redis_url: str, key: str) -> None:
    publish(redis_url, key, "submission", "intake_submission")
