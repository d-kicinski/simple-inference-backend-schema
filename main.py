from __future__ import annotations

import time

from demo_runtime import DemoController, format_result


def main() -> None:
    controller = DemoController()
    controller.start()

    while True:
        for result in controller.drain_results():
            print(f"[console] {format_result(result)}")

        if controller.finished.is_set() and not controller.is_running():
            return

        time.sleep(0.1)


if __name__ == "__main__":
    main()
