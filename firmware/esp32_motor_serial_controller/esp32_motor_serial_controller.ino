#include <Arduino.h>

const int ENA_PIN = 25;
const int IN1_PIN = 26;
const int IN2_PIN = 27;

const int PWM_FREQ = 5000;
const int PWM_RESOLUTION = 8;

const unsigned long COMMAND_TIMEOUT_MS = 1500;

unsigned long lastCommandMs = 0;
bool motorRunning = false;

void stopMotor() {
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  ledcWrite(ENA_PIN, 0);
  motorRunning = false;
}

void forwardMotor(int pwm) {
  pwm = constrain(pwm, 0, 255);

  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  ledcWrite(ENA_PIN, pwm);

  motorRunning = true;
  lastCommandMs = millis();
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);

  ledcAttach(ENA_PIN, PWM_FREQ, PWM_RESOLUTION);

  stopMotor();

  Serial.println("ESP32 motor serial controller ready");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("FWD")) {
      int spaceIndex = command.indexOf(' ');
      int pwm = 120;

      if (spaceIndex > 0) {
        pwm = command.substring(spaceIndex + 1).toInt();
      }

      forwardMotor(pwm);
      Serial.print("OK FWD ");
      Serial.println(pwm);
    }
    else if (command == "STOP") {
      stopMotor();
      lastCommandMs = millis();
      Serial.println("OK STOP");
    }
  }

  if (motorRunning && (millis() - lastCommandMs > COMMAND_TIMEOUT_MS)) {
    stopMotor();
    Serial.println("SAFETY STOP");
  }
}