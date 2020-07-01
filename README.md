# vtol-tiltwing
A project with the objective of autonomously taking off and hovering a tiltwing aircraft with the capability of transitioning into fixed-wing flight. Applications include search and rescue, firefighting, and military transport.

## Project Structure

### Overall Structure
The *header* files are located in _include/_ and the *source* files are located in _src_, and their filepaths match up.
The _lib_ folder contains external libraries that are linked in the project, including our *Logger* package, which Jason is very proud of.

### Source File Structure

- _src/_
    - _config.json_: A configuration file where all initializing variables are stored.
    - _mcl/_
        - _Supervisor_: The handler for all *ReadTasks*, *ControlTasks*, and *ActuateTasks*; it runs the MCL loop.
        - _Field_: The base class used in _Registry_ and _Flag_ to allow for all data types.
        - _Registry_: Includes all variables related to the state of the plane, as well as when they were last updated. This should be updated by the *ReadTasks*
        - _Flag_: Includes all commands that the *ActuateTasks* need to perform, should be reset every MCL cycle.
        - _ClockManager_: Used to keep the timing consistent in the MCL loop.
    - _read_tasks/_
        - _IMUReadTask_: Updates *Registry* with IMU data.
        - _GPSReadTask_: Updates *Registry* with GPS data.
        - _TelemetryReadTask_: Updates *Registry* with messages received via telemetry.
    - _control_tasks/_
        - _TelemetryControlTask_: Ingests all methods received via Telemetry
        - _FlightControlTask_: Determines an ideal flight path to reach a given destination. Also sets the mode (takeoff, cruise, landing).
        - _PIDControlTask_: Makes sure the plane stays upright and actually controls the motors to reach a given target orientation. For example, the Flight CT would determine that we need to "roll CW 30 degrees" and the PID CT would figure out the motor configuration to do that.
    - _actuate_tasks/_
        - _MotorActuateTask_: Actuates the motors based on target values set in *Flag*
        - _ServoActuateTask_: Actuates the servos based on target values set in *Flag*
    - _drivers/_: Contain the low-level code.
        - _TelemetryDriver_: Driver for the *Telemetry* module (need to figure out what we're using for telemetry).
        - _MotorDriver_: Driver for the brushless motors / ESCs.
        - _ServoDriver_: Driver for the servos.
        - _IMUDriver_: Driver for the BNO055 module.


