const uint8_t RLED {7U};  // 빨간 LED
const uint8_t GLED {8U};  // 초록 LED
const uint8_t Buzzer {3U}; // 부저 핀

int inputPin = 9; // PIR 센서 핀
int pirState = LOW;
int val = 0;
int hzData[8] ={262, 294, 330, 349, 392, 440 ,494, 523};

void setup() {
  pinMode(RLED, OUTPUT);
  pinMode(GLED, OUTPUT);
  pinMode(Buzzer, OUTPUT);
  pinMode(inputPin, INPUT);
  Serial.begin(115200UL);
  Serial1.begin(9600UL);
}

void loop() {

  val = digitalRead(inputPin);

  if (Serial.available() > 0) {
    String data = Serial.readString();  // 개행 문자 기준으로 데이터 읽기

    if (data.equals("ON\n")) {
      digitalWrite(GLED, HIGH);  // 초록 LED ON
      digitalWrite(RLED, LOW);   // 빨간 LED OFF
      tone(Buzzer, hzData[7], 100);   // 높은음
      Serial.println("맞습니다.");
      delay(10);
    } 
    else if (data.equals("OFF\n")) {
      digitalWrite(GLED, LOW);   // 초록 LED OFF
      digitalWrite(RLED, HIGH);  // 빨간 LED ON
      tone(Buzzer, hzData[0], 100);    // 낮은음
      Serial.println("틀립니다.");
      delay(10);
    }
  }

  if (val == HIGH) {
    Serial.println("TAKE_PHOTO");  // PIR 센서 감지 시 촬영 신호 전송
    delay(2000);
  } else {
    Serial.println("Undetected");
  }

  delay(1000);
}

