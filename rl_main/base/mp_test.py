#!/usr/bin/env python3
import time
import torch.multiprocessing as mp
from collections import deque

mp.set_start_method('spawn', force=True)


def play_func(exp_queue):
    step_idx = 0

    while True:
        step_idx += 1
        exp = step_idx

        exp_queue.put(exp)

        print(step_idx)

        if step_idx >= 1000:
            break

    exp_queue.put(None)


def main():
    buffer = deque(maxlen=1000)

    train_freq = 4
    replay_initial = 100

    exp_queue = mp.Queue(maxsize=train_freq * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue,))
    play_proc.start()

    while play_proc.is_alive():
        for _ in range(train_freq):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer.append(exp)

        if len(buffer) < replay_initial:
            continue

        batch = buffer.popleft()
        print(batch, " ----------------- ")
        time.sleep(1)


if __name__ == "__main__":
    main()