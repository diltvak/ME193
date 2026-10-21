// Cytron Maker Drive: two DC motors, each driven by two PWM inputs.
// Motor 1 (left): pins 5 & 6   Motor 2 (right): pins 10 & 11
// Per motor: (PWM, LOW) = forward, (LOW, PWM) = backward, (LOW, LOW) = stop.
//
// The Python behavior tree calls drive(left, right) over the Bridge with
// signed speeds in -255..255. A watchdog stops both motors if no drive
// command arrives for WATCHDOG_MS (Python crashed, App stopped, BLE stalled).

#include "Arduino_RouterBridge.h"
#include <Wire.h>

const int M1A = 5;
const int M1B = 6;
const int M2A = 10;
const int M2B = 11;

// Flip to -1 if a wheel spins the wrong way for "forward".
const int LEFT_SIGN = 1;
const int RIGHT_SIGN = 1;

const unsigned long WATCHDOG_MS = 300;

// MPU6050 on the A4 (SDA/PC1) / A5 (SCL/PC0) header pins = Wire2 on UNO Q (Wire = internal i2c2, Wire1 = Qwiic).
#define IMU_WIRE Wire2
const uint8_t MPU_ADDR = 0x68;  // 0x69 if AD0 is pulled high
const float ACCEL_LSB_PER_G = 16384.0;  // default +/-2 g range
bool imuReady = false;

unsigned long lastCommandMs = 0;
bool stopped = true;

void setMotor(int pinA, int pinB, int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0) {
    analogWrite(pinA, speed);
    analogWrite(pinB, 0);
  } else if (speed < 0) {
    analogWrite(pinA, 0);
    analogWrite(pinB, -speed);
  } else {
    analogWrite(pinA, 0);
    analogWrite(pinB, 0);
  }
}

void applyDrive(int left, int right) {
  setMotor(M1A, M1B, LEFT_SIGN * left);
  setMotor(M2A, M2B, RIGHT_SIGN * right);
}

// Bridge handler: called from Python with Bridge.notify("drive", left, right).
void drive(int left, int right) {
  applyDrive(left, right);
  lastCommandMs = millis();
  stopped = (left == 0 && right == 0);
}

// Bridge handler: x-axis acceleration in g (NaN if the MPU6050 is not responding).
float get_accel_x() {
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(0x3B);  // ACCEL_XOUT_H
  if (IMU_WIRE.endTransmission(false) != 0 || IMU_WIRE.requestFrom(MPU_ADDR, (uint8_t)2) != 2) {
    return NAN;
  }
  int16_t raw = (IMU_WIRE.read() << 8) | IMU_WIRE.read();
  return raw / ACCEL_LSB_PER_G;
}

void setup() {
  Serial.begin(115200);
  pinMode(M1A, OUTPUT);
  pinMode(M1B, OUTPUT);
  pinMode(M2A, OUTPUT);
  pinMode(M2B, OUTPUT);
  applyDrive(0, 0);

  IMU_WIRE.begin();
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(0x6B);  // PWR_MGMT_1: clear sleep bit
  IMU_WIRE.write(0);
  imuReady = IMU_WIRE.endTransmission() == 0;
  Serial.println(imuReady ? "MPU6050 ready" : "MPU6050 not found");

  Bridge.begin();
  // provide_safe runs the handler in the loop() context, so it never races the watchdog.
  Bridge.provide_safe("drive", drive);
  Bridge.provide_safe("get_accel_x", get_accel_x);
  Serial.println("lego_robot sketch ready");
}

void loop() {
  if (!stopped && millis() - lastCommandMs > WATCHDOG_MS) {
    applyDrive(0, 0);
    stopped = true;
    Serial.println("Watchdog: no drive command, motors stopped");
  }
  delay(5);
}
