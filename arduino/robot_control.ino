/*
 * robot_control.ino
 * =================
 * Arduino sketch for the Autonomous Weed Removal Robot.
 *
 * What this sketch does
 * ---------------------
 * 1. Listens for single-byte commands on the hardware Serial port
 *    (USB cable from laptop running the Python vision code).
 * 2. Executes the corresponding motor/blade action.
 * 3. Sends a status string back to the laptop for confirmation.
 *
 * Serial protocol (9600 baud, 8-N-1)
 * ------------------------------------
 *   Receive 'M'  →  MOVE  – enable drive motors, disable blade
 *   Receive 'S'  →  STOP  – disable all motors
 *   Receive 'C'  →  CUT   – disable drive motors, enable blade motor
 *
 * Pin mapping  (adjust to your actual wiring)
 * -------------------------------------------
 *   PIN_DRIVE_EN   2   – L298N / motor driver ENABLE for drive motors
 *   PIN_DRIVE_IN1  3   – motor direction IN1
 *   PIN_DRIVE_IN2  4   – motor direction IN2
 *   PIN_BLADE_EN   5   – BLDC motor driver ENABLE
 *   PIN_LED_STATUS 13  – built-in LED (blinks on command)
 *
 * Connections
 * -----------
 *   Laptop USB  ──►  Arduino USB (Serial)
 *   Arduino D2  ──►  L298N ENA
 *   Arduino D3  ──►  L298N IN1
 *   Arduino D4  ──►  L298N IN2
 *   Arduino D5  ──►  BLDC driver Enable
 *
 * Power supply
 * ------------
 *   Drive motors / BLDC:  48 V external supply through motor drivers
 *   Arduino:              powered via USB from laptop
 *
 * Note: Keep 48 V circuitry isolated from Arduino logic pins.
 */

// ── pin definitions ──────────────────────────────────────────────────────────
const int PIN_DRIVE_EN  = 2;
const int PIN_DRIVE_IN1 = 3;
const int PIN_DRIVE_IN2 = 4;
const int PIN_BLADE_EN  = 5;
const int PIN_LED       = 13;

// ── state ────────────────────────────────────────────────────────────────────
char currentCommand = 'S';   // start in STOP state

// ── helpers ──────────────────────────────────────────────────────────────────

void driveMotors(bool enabled) {
  digitalWrite(PIN_DRIVE_EN, enabled ? HIGH : LOW);
  if (enabled) {
    // Forward direction
    digitalWrite(PIN_DRIVE_IN1, HIGH);
    digitalWrite(PIN_DRIVE_IN2, LOW);
  } else {
    digitalWrite(PIN_DRIVE_IN1, LOW);
    digitalWrite(PIN_DRIVE_IN2, LOW);
  }
}

void bladeMotor(bool enabled) {
  digitalWrite(PIN_BLADE_EN, enabled ? HIGH : LOW);
}

void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(PIN_LED, HIGH);
    delay(delayMs);
    digitalWrite(PIN_LED, LOW);
    delay(delayMs);
  }
}

// ── command handlers ─────────────────────────────────────────────────────────

void doMove() {
  driveMotors(true);
  bladeMotor(false);
  digitalWrite(PIN_LED, HIGH);
  Serial.println("STATUS: MOVING");
}

void doStop() {
  driveMotors(false);
  bladeMotor(false);
  blinkLED(2, 150);
  Serial.println("STATUS: STOPPED");
}

void doCut() {
  driveMotors(false);
  bladeMotor(true);
  blinkLED(3, 100);
  Serial.println("STATUS: CUTTING");
}

// ── Arduino setup ─────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(9600);

  pinMode(PIN_DRIVE_EN,  OUTPUT);
  pinMode(PIN_DRIVE_IN1, OUTPUT);
  pinMode(PIN_DRIVE_IN2, OUTPUT);
  pinMode(PIN_BLADE_EN,  OUTPUT);
  pinMode(PIN_LED,       OUTPUT);

  // Start in STOP state
  doStop();
  Serial.println("READY – Weed Removal Robot");
  Serial.println("Waiting for commands: M=MOVE  S=STOP  C=CUT");
}

// ── Arduino loop ──────────────────────────────────────────────────────────────

void loop() {
  if (Serial.available() > 0) {
    char cmd = (char)Serial.read();

    if (cmd == currentCommand) {
      return;   // already in this state – skip
    }

    currentCommand = cmd;

    switch (cmd) {
      case 'M': doMove(); break;
      case 'S': doStop(); break;
      case 'C': doCut();  break;
      default:
        Serial.print("UNKNOWN CMD: ");
        Serial.println(cmd);
        doStop();
        break;
    }
  }
}
