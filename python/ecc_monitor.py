"""Background ECC poller. Run one instance per node.

Logs corrected/uncorrected ECC counters, temperature, power, and SM clock
every 10s into /tmp/ecc_<host>.log so you can correlate any errors with the
training iteration timestamps printed by main.py.
"""
from __future__ import annotations
import datetime
import socket
import subprocess
import sys
import time

INTERVAL = 10
QUERY = ("index,ecc.errors.corrected.volatile.total,"
         "ecc.errors.uncorrected.volatile.total,"
         "temperature.gpu,power.draw,clocks.sm")

def main():
    host = socket.gethostname()
    log = open(f"/tmp/ecc_{host}.log", "a", buffering=1)
    log.write(f"# started {datetime.datetime.now().isoformat()} on {host}\n")
    while True:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={QUERY}",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.STDOUT, timeout=8).decode()
            for line in out.strip().splitlines():
                log.write(f"{ts} {host} {line}\n")
        except Exception as e:                      # noqa: BLE001
            log.write(f"{ts} {host} ERROR {e}\n")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
