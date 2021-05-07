#include <vtol_tiltwing/IMUReadTask.hpp>

void IMUReadTask::execute(Registry *registry){
    // Print calibration status
    uint8_t system, gyro, accel, mag = 0;
    bno.getCalibration(&system, &gyro, &accel, &mag);
    Logger::log("Calibration: sys=" + to_string(system) + ", gyro=" + to_string(gyro) + ", accel=" + to_string(accel) + ", mag=" + to_string(mag), LogSeverity::DEBUG);

    // Get sensor data
    sensors_event_t orientationData;
    bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
    float x = orientationData.orientation.x;
    float y = orientationData.orientation.y;
    float z = orientationData.orientation.z;
    Logger::log("Orientation: x=" + to_string(system) + ", y=" + to_string(gyro) + ", z=" + to_string(accel), LogSeverity::INFO);

    // TODO: This is a guess, need to figure out which axis is which
    registry->put("imu.pitch", x);
    registry->put("imu.roll", y);
    registry->put("imu.yaw", z);
}

