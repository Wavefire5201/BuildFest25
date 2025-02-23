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
    extracted_text_result = extract_text_from_image(image_path)

    if not extracted_text_result:
        print("No text extracted.")
        return

    print("Extracted Text:\n", extracted_text_result)

    chunks = chunk_text(extracted_text_result.extracted_text)
    for i, chunk in enumerate(chunks, start=1):
        sentiment_data = analyze_mood(chunk)
        if sentiment_data.temperature is not None and sentiment_data.rgb is not None:
            print(f"Chunk {i}: {chunk}")
            print(
                f"Temperature: {sentiment_data.temperature}, RGB: {sentiment_data.rgb}"
            )
            print(sentiment_data)
            for dev in devices:
                dev.activate_thermal_intensity_control(int(sentiment_data.temperature))
                dev.play_frequency(float(sentiment_data.vibration_frequency), 1.0)
                dev.set_led(
                    sentiment_data.rgb.red,
                    sentiment_data.rgb.green,
                    sentiment_data.rgb.blue,
                )

            output_filename = f"{i}.mp3"
            kokoro_tts(chunk, output_filename)
    for dev in devices:
        dev.disable_all_thermal()


if __name__ == "__main__":
    main()
