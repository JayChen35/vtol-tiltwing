#include <Servo.h>

Servo motor;

const int MIN_VAL = 1000;
const int MAX_VAL = 2000;
const int TEST_VAL = 1100;

void setup() {
  // put your setup code here, to run once:
  motor.attach(9);

  Serial.begin(9600);
  Serial.println("Arming");
  arm();
//  calibrate();
  delay(2000);
  Serial.println("Go for it");
  motor.writeMicroseconds(MIN_VAL);
}

void calibrate(){
  motor.writeMicroseconds(0);
  Serial.println("Disconnect");
  delay(3000);
  motor.writeMicroseconds(MAX_VAL);
  Serial.println("Connect");
  delay(3000);
  motor.writeMicroseconds(MIN_VAL);
  delay(6000);
  motor.writeMicroseconds(0);
  delay(2000);
  motor.writeMicroseconds(MIN_VAL);  
}

void arm(){
  // Arming?
  motor.writeMicroseconds(0);
  Serial.println("Connect");
  delay(3000);
  motor.writeMicroseconds(0);
  delay(2000);
  motor.writeMicroseconds(MAX_VAL);
  delay(2000);
  motor.writeMicroseconds(MIN_VAL);
}

void test(){
  int sped = 1200;
  motor.writeMicroseconds(TEST_VAL);
  Serial.println("Waiting");
  delay(250);
  Serial.println("Done waiting");
  motor.writeMicroseconds(MIN_VAL);
}

void loop() {
  // put your main code here, to run repeatedly:
//  Serial.println("Go for it");
  while(Serial.available()){
    int c = Serial.read();
    if(c < 49 || c > 57){
      continue;
    }
    test();
    /*
    int sped = map(c, 49, 57, 1000, 2000);
    Serial.println(sped);
    motor.writeMicroseconds(sped);
    */
  }
//  motor.writeMicroseconds(TEST_VAL);
//  Serial.println("Testing");
//  delay(2000);
//  motor.writeMicroseconds(MIN_VAL);
//  Serial.println("Stopping");
//  delay(3000);
  delay(10);
}
