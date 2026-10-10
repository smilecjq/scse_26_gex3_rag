import os
import subprocess
import time


def wait_for_gpu():
    if os.environ.get("POLICY_THERMAL_GUARD") != "1":
        return
    start = time.monotonic()
    cooling = False
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        temperature = float(result.stdout.strip())
        if temperature >= 80:
            raise RuntimeError("GPU temperature reached the stop limit")
        if temperature < 72 and not cooling:
            return
        if temperature <= 65:
            return
        if not cooling:
            print(f"Waiting for GPU to cool: {temperature:.0f} C", flush=True)
        cooling = True
        if time.monotonic() - start > 180:
            raise RuntimeError("GPU did not cool within three minutes")
        time.sleep(2)
