#include "Arduino_RouterBridge.h"
#include "Arduino_LED_Matrix.h"

Arduino_LED_Matrix matrix;

const int COLS = 13;
const int ROWS = 8;

// 3x5 font, one byte per row (low 3 bits, MSB = left pixel). Digits 0-9.
const uint8_t DIGITS[10][5] = {
    {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
    {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7},
};
const uint8_t MINUS[5] = {0, 0, 7, 0, 0};

volatile int displayValue = 0;
volatile bool displayDirty = true;
volatile bool ringOn = true;  // no MQTT data yet at startup

// Draw a 3x5 glyph with its top-left corner at (col0, row0) into a packed frame.
void drawGlyph(uint32_t frame[4], const uint8_t glyph[5], int col0, int row0) {
    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < 3; c++) {
            if (glyph[r] & (1 << (2 - c))) {
                int i = (row0 + r) * COLS + (col0 + c);
                frame[i / 32] |= 1UL << (31 - (i % 32));
            }
        }
    }
}

// Called from Python. Shows -99..99 as sign + up to two digits (e.g. x * 100).
bool show_number(int value) {
    displayValue = constrain(value, -99, 99);
    displayDirty = true;
    return true;
}

// Called from Python. While on, the matrix shows only the outer ring.
bool show_ring(bool on) {
    ringOn = on;
    displayDirty = true;
    return true;
}

void renderRing() {
    uint32_t frame[4] = {0, 0, 0, 0};
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (r == 0 || r == ROWS - 1 || c == 0 || c == COLS - 1) {
                int i = r * COLS + c;
                frame[i / 32] |= 1UL << (31 - (i % 32));
            }
        }
    }
    matrix.loadFrame(frame);
}

void renderNumber() {
    if (ringOn) {
        renderRing();
        return;
    }
    int v = displayValue;
    int mag = abs(v);
    uint32_t frame[4] = {0, 0, 0, 0};
    // Three fixed 3-px slots with 1-px gaps (11 cols), centred: cols 1, 5, 9.
    if (v < 0) drawGlyph(frame, MINUS, 1, 1);
    if (mag >= 10) drawGlyph(frame, DIGITS[mag / 10], 5, 1);
    drawGlyph(frame, DIGITS[mag % 10], 9, 1);
    matrix.loadFrame(frame);
}

// Cytron Maker Drive: each motor has two inputs.
// (PWM, LOW) = one direction, (LOW, PWM) = the other, (LOW, LOW) = stop.
//
// NOTE: hardware PWM (analogWrite) hangs the MCU on this board/platform build
// (arduino:zephyr 0.52.0): the Bridge stops answering and every call times out.
// Pin 11 also has no PWM on the UNO Q. So PWM is done in software here with
// plain digitalWrite(), which works on any pin.
const int M1A = 5;   // left motor
const int M1B = 6;
const int M2A = 10;  // right motor
const int M2B = 11;

const unsigned long PWM_PERIOD_US = 2000;  // 500 Hz software PWM

// If Python stops sending commands, stop the motors after this long.
const unsigned long TIMEOUT_MS = 500;

volatile int leftCmd = 0;    // -255..255, negative = reverse
volatile int rightCmd = 0;
volatile unsigned long lastCommand = 0;

// Drive one motor's two pins for the current point in the PWM period.
void drivePins(int pinA, int pinB, int cmd, unsigned long phaseUs) {
    cmd = constrain(cmd, -255, 255);
    unsigned long dutyUs = (unsigned long)abs(cmd) * PWM_PERIOD_US / 255;
    bool on = phaseUs < dutyUs;
    digitalWrite(pinA, (cmd > 0 && on) ? HIGH : LOW);
    digitalWrite(pinB, (cmd < 0 && on) ? HIGH : LOW);
}

// Called from Python. left/right are -255..255 (negative = reverse).
bool set_motors(int left, int right) {
    leftCmd = left;
    rightCmd = right;
    lastCommand = millis();
    return true;
}

void setup() {
    Bridge.begin();
    Bridge.provide("set_motors", set_motors);
    Bridge.provide("show_number", show_number);
    Bridge.provide("show_ring", show_ring);
    matrix.begin();
    matrix.clear();
    pinMode(M1A, OUTPUT);
    pinMode(M1B, OUTPUT);
    pinMode(M2A, OUTPUT);
    pinMode(M2B, OUTPUT);
    lastCommand = millis();
}

void loop() {
    if (displayDirty) {
        displayDirty = false;
        renderNumber();
    }
    if (millis() - lastCommand > TIMEOUT_MS) {
        leftCmd = 0;
        rightCmd = 0;
    }
    unsigned long phase = micros() % PWM_PERIOD_US;
    drivePins(M1A, M1B, leftCmd, phase);
    drivePins(M2A, M2B, rightCmd, phase);
    delayMicroseconds(50);
}
