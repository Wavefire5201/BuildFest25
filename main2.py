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
audio_queue = Queue()

# Lock object for managing audio playback
audio_lock = threading.Lock()


def main():
    pygame.init()
    while True:
        scanning()
        image = capture_desktop_frame()
        if process_image_file(image):
            print("All pages processed. Resetting and starting over.")
            flip_page()
            reset_queues()


def process_image_file(page: str) -> bool:
    extracted_text_result = extract_text_from_image(page)

    if not extracted_text_result.has_book_in_image or not extracted_text_result:
        not_detected()
        print("No book detected in image")
        return False

    print("Extracted Text from:\n", extracted_text_result)

    # Use an Event object to signal thread completion
    generate_audio_done = threading.Event()
    play_audio_done = threading.Event()

    chunks = chunk_text(extracted_text_result.extracted_text)
    for index, chunk in enumerate(chunks, start=1):
        chunk_queue.put((index, chunk))

    def generate_audio_files_with_event():
        generate_audio_files()
        generate_audio_done.set()

    def play_audio_with_event():
        play_audio_from_queue()
        play_audio_done.set()

    # Start the threads and pass the event clearing task
    threading.Thread(target=generate_audio_files_with_event, daemon=True).start()
    threading.Thread(target=play_audio_with_event, daemon=True).start()

    # Wait for both threads to signal they're done
    generate_audio_done.wait()
    play_audio_done.wait()
    return True


def generate_audio_files():
    while not chunk_queue.empty():
        index, chunk = chunk_queue.get()
        run(index, chunk)
        chunk_queue.task_done()


def run(index, chunk):
    sentiment_data = analyze_mood(chunk)
    output_filename = f"{index}.mp3"
    kokoro_tts(chunk, output_filename)
    audio_queue.put((output_filename, sentiment_data, chunk))


def play_audio_from_queue():
    pygame.mixer.init()
    while True:
        if not audio_queue.empty():
            filename, sentiment_data, text_chunk = audio_queue.get()
            print(f"Playing: {text_chunk}")
            play_audio(filename)
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            update_devices_with_sentiment(sentiment_data)
            audio_queue.task_done()


def play_audio(filename):
    with audio_lock:
        pygame.mixer.init()
        pygame.mixer.music.load(f"./audio/{filename}")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)


def update_devices_with_sentiment(sentiment_data):
    if sentiment_data.temperature is not None and sentiment_data.rgb is not None:
        print(
            f"Updating devices with Temperature: {sentiment_data.temperature}, RGB: {sentiment_data.rgb}"
        )
        for dev in devices:
            dev.activate_thermal_intensity_control(int(sentiment_data.temperature))
            dev.set_led(
                sentiment_data.rgb.red,
                sentiment_data.rgb.green,
                sentiment_data.rgb.blue,
            )


def reset_queues():
    with chunk_queue.mutex:
        chunk_queue.queue.clear()
    with audio_queue.mutex:
        audio_queue.queue.clear()


def scanning():
    play_audio_with_lock("./audio_prompts/scanning.mp3")
    for dev in devices:
        dev.play_vibration_sequence([VibrationWaveforms.BUZZ1_P100])


def flip_page():
    play_audio_with_lock("./audio_prompts/flip.mp3")
    vibration_sequence = [
        VibrationWaveforms.STRONG_BUZZ_P100,
        VibrationWaveforms.Rest(0.5),
        VibrationWaveforms.TRANSITION_HUM1_P100,
        VibrationWaveforms.Rest(0.5),
        VibrationWaveforms.TRANSITION_RAMP_DOWN_MEDIUM_SHARP2_P50_TO_P0,
        VibrationWaveforms.TRANSITION_RAMP_UP_MEDIUM_SHARP2_P0_TO_P50,
        VibrationWaveforms.DOUBLE_CLICK_P100,
        VibrationWaveforms.TRANSITION_RAMP_UP_SHORT_SMOOTH2_P0_TO_P100,
    ]
    for dev in devices:
        dev.play_vibration_sequence(vibration_sequence)


def not_detected():
    play_audio_with_lock("./audio_prompts/not_detected.mp3")
    for dev in devices:
        dev.play_vibration_sequence(
            [VibrationWaveforms.LONG_DOUBLE_SHARP_CLICK_MEDIUM1_P100]
        )


def play_audio_with_lock(filename):
    with audio_lock:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.quit()  # Quit mixer after playing to clean up resources


if __name__ == "__main__":
    main()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        pygame.quit()
