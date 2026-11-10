"""Apply resource limits in a fresh process, safely before spawning Git."""

import os
import resource
import sys


def main() -> None:
    arguments = sys.argv[1:]
    if arguments[0] == "--coordinator":
        arguments = arguments[1:]
    else:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.execvpe(arguments[0], arguments, os.environ)


if __name__ == "__main__":
    main()
