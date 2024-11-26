from pynput import keyboard

pressed_keys = set()

def on_press(key):
    if key not in pressed_keys:
        pressed_keys.add(key)
        try:
            print(f'Key {key.char} pressed')
        except AttributeError:
            print(f'Special key {key} pressed')

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)
    print(f'Key {key} released')
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()