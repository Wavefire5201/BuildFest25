import time
import threading
from queue import Queue
import pygame
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

chunk_queue = Queue()


def main():
    # allow time to switch to the window
    time.sleep(5)

    # page = image_to_base64("./IMG20250223032711.jpg")
    page = frame_to_base64(capture_desktop_frame())
    extracted_text_result = extract_text_from_image(page)

    if not extracted_text_result:
        print("No text extracted.")
        return

    print("Extracted Text:\n", extracted_text_result)

    chunks = chunk_text(extracted_text_result.extracted_text)
    for index, chunk in enumerate(chunks, start=1):
        chunk_queue.put((index, chunk))

    audio_thread = threading.Thread(target=play_chunks_from_queue)
    audio_thread.start()
    # while True:
    #     run(page)
    for dev in devices:
        dev.disable_all_thermal()


def run(index, chunk):
    # extracted_text_result = extract_text_from_image(image_path)

    # if not extracted_text_result:
    #     print("No text extracted.")
    #     return

    # print("Extracted Text:\n", extracted_text_result)

    # chunks = chunk_text(extracted_text_result.extracted_text)
    # for i, chunk in enumerate(chunks, start=1):
    sentiment_data = analyze_mood(chunk)
    if sentiment_data.temperature is not None and sentiment_data.rgb is not None:
        print(f"Chunk {index}: {chunk}")
        print(f"Temperature: {sentiment_data.temperature}, RGB: {sentiment_data.rgb}")
        print(sentiment_data)
        for dev in devices:
            dev.activate_thermal_intensity_control(int(sentiment_data.temperature))
            # dev.play_frequency(float(sentiment_data.vibration_frequency), 1.0)
            dev.set_led(
                sentiment_data.rgb.red,
                sentiment_data.rgb.green,
                sentiment_data.rgb.blue,
            )

        output_filename = f"{index}.mp3"
        kokoro_tts(chunk, output_filename)

        play_audio(output_filename)


def play_chunks_from_queue():
    while not chunk_queue.empty():
        index, chunk = chunk_queue.get()
        run(index, chunk)
        chunk_queue.task_done()


def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(f"./audio/{filename}")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue


if __name__ == "__main__":
    main()
