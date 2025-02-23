import time
from inference import *
from utils import *
from datafeel.device import (
    VibrationMode,
    discover_devices,
    LedMode,
    ThermalMode,
    VibrationWaveforms,
)

devices = discover_devices(4)

device = devices[0]

print("found", len(devices), "devices")
print(device)
print("Reading all data...")
print("skin temperature:", device.registers.get_skin_temperature())
print("sink temperature:", device.registers.get_sink_temperature())
print("mcu temperature:", device.registers.get_mcu_temperature())
print("gate driver temperature:", device.registers.get_gate_driver_temperature())
print("thermal power:", device.registers.get_thermal_power())
print("thermal mode:", device.registers.get_thermal_mode())
print("thermal intensity:", device.registers.get_thermal_intensity())
print("vibration mode:", device.registers.get_vibration_mode())
print("vibration frequency:", device.registers.get_vibration_frequency())
print("vibration intensity:", device.registers.get_vibration_intensity())
print("vibration go:", device.registers.get_vibration_go())
print("vibration sequence 0123:", device.registers.get_vibration_sequence_0123())
print("vibration sequence 3456:", device.registers.get_vibration_sequence_3456())
device.registers.set_vibration_mode(VibrationMode.MANUAL)


def main():
    # allow time to switch to the window
    # time.sleep(5)
    # frame = capture_desktop_frame()
    frame = image_to_base64("./IMG20250223032711.jpg")
    # run(frame_to_base64(frame))
    run(frame)


def run(image_path):
    result = extract_text_from_image(image_path)
    print(result)
    print(type(result))

    if not result:
        print("No text extracted.")
        return

    print("Extracted Text:\n", result)

    chunks = chunk_text(result["extracted_text"])
    for idx, chunk in enumerate(chunks, start=1):
        res = analyze_mood(chunk)
        if res["temperature"] is not None and res["rgb"] is not None:
            print(f"Chunk {idx}: {chunk}")
            print(f"Temperature: {res['temperature']}, RGB: {res['rgb']}")
            print(res)
            device.activate_thermal_intensity_control(int(res["temperature"]))
            device.play_frequency(float(res["vibration_frequency"]), 1.0)
            device.set_led(
                int(res["rgb"]["red"]),
                int(res["rgb"]["green"]),
                int(res["rgb"]["blue"]),
            )

            output_filename = f"chunk_{idx}.mp3"
            convert_text_to_speech(chunk, output_filename)
    device.disable_all_thermal()


if __name__ == "__main__":
    main()
