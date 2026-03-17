"""
robot_controller.py
===================
Simulated robot controller.

What this module does
---------------------
Receives a command string (MOVE / STOP / CUT) from the vision system and:

  1. Prints a human-readable status to the terminal.
  2. Logs every command to  robot_log.txt  with a timestamp.
  3. (Optional) sends a single ASCII byte over a serial port to a real
     Arduino – enabled automatically when  serial_port  is provided.

Serial protocol (for Arduino integration)
------------------------------------------
  'M'  →  MOVE  – motors running, blade off
  'S'  →  STOP  – all motors off
  'C'  →  CUT   – blade motor on, drive motors off

Arduino sketch: see  arduino/robot_control.ino
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

# ── logging setup ──────────────────────────────────────────────────────────
LOG_FILE = Path("robot_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("RobotController")

# ── command byte map ──────────────────────────────────────────────────────
_CMD_BYTES = {
    "MOVE": b"M",
    "STOP": b"S",
    "CUT":  b"C",
}

_EMOJI = {
    "MOVE": "🟢 MOVE  – robot moving forward",
    "STOP": "🔴 STOP  – weed detected, robot stopped",
    "CUT":  "⚡ CUT   – cutting blade activated",
}


class RobotController:
    """
    Simulated (and optionally real) robot command dispatcher.

    Parameters
    ----------
    serial_port : str | None
        Serial port to use for Arduino communication (e.g. '/dev/ttyUSB0'
        on Linux, 'COM3' on Windows).  Leave  None  for simulation-only.
    baud_rate : int
        Baud rate for serial communication (must match Arduino sketch).
    debounce_s : float
        Minimum seconds between identical consecutive commands.  Prevents
        flooding the serial port / log when the scene is static.
    """

    def __init__(
        self,
        serial_port: str | None = None,
        baud_rate: int = 9600,
        debounce_s: float = 0.5,
    ):
        self.debounce_s = debounce_s
        self._last_cmd: str = ""
        self._last_time: float = 0.0
        self._ser = None

        if serial_port:
            self._open_serial(serial_port, baud_rate)

    # ── serial helpers ────────────────────────────────────────────────────

    def _open_serial(self, port: str, baud: int):
        try:
            import serial  # noqa: PLC0415
            self._ser = serial.Serial(port, baud, timeout=1)
            logger.info("Serial port %s opened at %d baud", port, baud)
        except ImportError:
            logger.warning("pyserial not installed – serial output disabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open serial port %s: %s", port, exc)

    def _send_serial(self, cmd: str):
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(_CMD_BYTES.get(cmd, b"S"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Serial write failed: %s", exc)

    # ── public API ────────────────────────────────────────────────────────

    def execute(self, command: str):
        """
        Dispatch a robot command.

        Parameters
        ----------
        command : str
            One of  'MOVE',  'STOP',  or  'CUT'.
        """
        command = command.upper().strip()
        if command not in _CMD_BYTES:
            logger.warning("Unknown command '%s' – defaulting to STOP", command)
            command = "STOP"

        now = time.monotonic()
        # Debounce – skip if same command sent very recently
        if command == self._last_cmd and (now - self._last_time) < self.debounce_s:
            return

        self._last_cmd = command
        self._last_time = now

        # Terminal output
        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}]  {_EMOJI[command]}")

        # File log
        logger.info("Command: %s", command)

        # Arduino serial
        self._send_serial(command)

    def close(self):
        """Release serial port if open."""
        if self._ser and self._ser.is_open:
            self._ser.close()
            logger.info("Serial port closed")

    def __del__(self):
        self.close()
