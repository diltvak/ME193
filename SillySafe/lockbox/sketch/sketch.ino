// Finger Lockbox - microcontroller side.
//
// Wiring:
//   Button : D2 <-> GND   (uses the internal pull-up, no resistor needed)
//   Servo  : signal D9, power 5V, ground GND
//
// Python (python/main.py) polls take_button(), and calls set_servo() and show().

#include "Arduino_RouterBridge.h"
#include "Arduino_LED_Matrix.h"

const int BUTTON_PIN = 2;
const int SERVO_PIN = 9;

// The servo's fixed range of motion. Change these two angles to fit your latch.
const int LOCKED_ANGLE = 0;
const int OPEN_ANGLE = 90;

// The servo is driven only while it moves, then the signal stops so it sits
// still and quiet. Pulses are made with digitalWrite() because hardware PWM
// (analogWrite) hangs the MCU on this board's platform build.
const unsigned long MOVE_MS = 800;
const unsigned long DEBOUNCE_MS = 40;

Arduino_LED_Matrix matrix;
const int COLS = 13;

volatile bool pressed = false;          // latched until Python reads it
volatile int servoAngle = LOCKED_ANGLE;
volatile unsigned long moveStart = 0;
volatile bool moving = true;            // drive to LOCKED_ANGLE at power-up
volatile int displayCode = 10;
volatile bool displayDirty = true;

// 3x5 digits and 5x5 icons, one byte per row (MSB of the glyph width = left pixel).
const uint8_t DIGITS[10][5] = {
    {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
    {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7},
};
const uint8_t QUESTION[5] = {7, 1, 2, 0, 2};
const uint8_t CHECK[5] = {1, 2, 20, 8, 0};
const uint8_t CROSS[5] = {17, 10, 4, 10, 17};

void drawGlyph(uint32_t frame[4], const uint8_t glyph[5], int width, int col0, int row0) {
    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < width; c++) {
            if (glyph[r] & (1 << (width - 1 - c))) {
                int i = (row0 + r) * COLS + (col0 + c);
                frame[i / 32] |= 1UL << (31 - (i % 32));
            }
        }
    }
}

void render() {
    uint32_t frame[4] = {0, 0, 0, 0};
    int code = displayCode;
    if (code >= 0 && code <= 9) drawGlyph(frame, DIGITS[code], 3, 5, 1);
    else if (code == 11) drawGlyph(frame, QUESTION, 3, 5, 1);
    else if (code == 12) drawGlyph(frame, CHECK, 5, 4, 1);
    else if (code == 13) drawGlyph(frame, CROSS, 5, 4, 1);
    matrix.loadFrame(frame);   // code 10 (or anything else) = blank
}

// --- Functions called from Python ---------------------------------------

// True once for each button press.
bool take_button() {
    bool was = pressed;
    pressed = false;
    return was;
}

// open = true moves to OPEN_ANGLE, false to LOCKED_ANGLE.
bool set_servo(bool open) {
    servoAngle = open ? OPEN_ANGLE : LOCKED_ANGLE;
    moveStart = millis();
    moving = true;
    return true;
}

// 0-9 = digit, 10 = blank, 11 = "?", 12 = check mark, 13 = X.
bool show(int code) {
    displayCode = code;
    displayDirty = true;
    return true;
}

// -------------------------------------------------------------------------

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(SERVO_PIN, OUTPUT);
    digitalWrite(SERVO_PIN, LOW);
    matrix.begin();
    matrix.clear();
    Bridge.begin();
    Bridge.provide("take_button", take_button);
    Bridge.provide("set_servo", set_servo);
    Bridge.provide("show", show);
    moveStart = millis();
}

void loop() {
    // Button: register a press on a debounced HIGH -> LOW edge.
    static bool stable = HIGH, lastRead = HIGH;
    static unsigned long lastChange = 0;
    bool reading = digitalRead(BUTTON_PIN);
    if (reading != lastRead) {
        lastRead = reading;
        lastChange = millis();
    }
    if (reading != stable && millis() - lastChange > DEBOUNCE_MS) {
        stable = reading;
        if (stable == LOW) pressed = true;
    }

    if (displayDirty) {
        displayDirty = false;
        render();
    }

    // Servo: one 50 Hz pulse per pass while moving (0 deg = 544 us, 180 deg = 2400 us).
    if (moving) {
        if (millis() - moveStart > MOVE_MS) {
            moving = false;
        } else {
            unsigned int us = 544 + (unsigned long)constrain(servoAngle, 0, 180) * (2400 - 544) / 180;
            digitalWrite(SERVO_PIN, HIGH);
            delayMicroseconds(us);
            digitalWrite(SERVO_PIN, LOW);
            delay(18);
            return;
        }
    }
    delay(2);
}
