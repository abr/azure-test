# Setup

1. Set up an Azure VM, following these instructions: 
   https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal
   
2. Set up an Azure IoT Hub/Device, following these instructions:
   https://docs.microsoft.com/en-us/azure/iot-hub/quickstart-send-telemetry-python
   
   - note down the device connection string, which looks like
     `HostName={YourIoTHubName}.azure-devices.net;DeviceId=MyPythonDevice;SharedAccessKey={YourSharedAccessKey}`,
     as you will need it later
     
3. Use `ssh` to connect to the Azure VM (in the same way as step 1 above,
   in the section "Connect to virtual machine")

4. Run `git clone https://github.com/abr/azure-test`

6. Run `cd azure-test`

5. Run `bash setup.sh`

6. Begin monitoring for events in your Azure IoT hub (following the same steps
   as in step 2 above, in the section "Read the telemetry from your hub")

6. Run `python nengo_test.py "<connection string>"`, where
   `<connection string>` is the string you noted down in step 2 above
   
You will see output that looks like:
```
Ground truth: 7
Detected digit: 7
Ground truth: 2
Detected digit: 2
...
```

In the Azure IoT hub you will see messages received that look like
```
{
    "event": {
        "origin": "nengo-device",
        "payload": "Detected digit: 7"
    }
}
{
    "event": {
        "origin": "nengo-device",
        "payload": "Detected digit: 2"
    }
}
...
```
