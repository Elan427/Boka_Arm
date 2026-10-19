// SO-101 gripper touch sensing: two flex sensors, one per jaw.
//
// Wiring (each sensor): 3.3V --- flex sensor --- node --- 10k --- GND, with the
// node going to the analog pin. A0 = fixed jaw sensor, A1 = moving jaw sensor.
// Bending (pressing) a flex sensor raises its resistance, which LOWERS the node
// voltage in this arrangement -- the host script handles the direction itself, it
// tares the free-air baseline at start-up and treats any move away from it as touch.
//
// Mega/Uno ADCs reference 5V by default, so a 3.3V divider reads at most ~675 of
// 1023. That is fine; do NOT call analogReference(EXTERNAL) unless AREF is wired
// to 3.3V.
//
// Output, 115200 baud, 240 lines/s (= 4 samples per 60 Hz mains cycle, so a 4-sample
// boxcar on the host cancels mains hum; use 200 Hz = 5/cycle on 50 Hz mains):   <A0>,<A1>

// Each value is the mean of OVERSAMPLE conversions.

const uint8_t PIN_A = A0;
const uint8_t PIN_B = A1;
const uint8_t OVERSAMPLE = 8;
const unsigned long PERIOD_US = 4167;  // 240 Hz

unsigned long next_us;

static uint16_t read_avg(uint8_t pin) {
  uint32_t sum = 0;
  for (uint8_t i = 0; i < OVERSAMPLE; i++) sum += analogRead(pin);
  return sum / OVERSAMPLE;
}

void setup() {
  Serial.begin(115200);
  next_us = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - next_us) < 0) return;
  next_us += PERIOD_US;
  if ((long)(now - next_us) > (long)PERIOD_US * 4) next_us = now + PERIOD_US;  // fell behind
  uint16_t a = read_avg(PIN_A);
  uint16_t b = read_avg(PIN_B);
  Serial.print(a);
  Serial.print(',');
  Serial.println(b);
}
