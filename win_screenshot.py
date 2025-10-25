import pygetwindow as gw
import os
from PIL import ImageGrab
import time

# --- Konfiguration ---
# Passen Sie diesen Titel an den exakten Titel des Spielfensters an.
# Beispiele: "BlueStacks App Player", "LastWar: Survival Game"
WINDOW_TITLE = "Last War-Survival Game" 
# Der Ordner, in dem die Screenshots gespeichert werden sollen.
OUTPUT_FOLDER = "screenshots"

def create_screenshot_from_window(window_title: str, output_folder: str):
    """
    Sucht ein Fenster nach seinem Titel, erstellt einen Screenshot davon und speichert ihn ab.

    Args:
        window_title (str): Der exakte Titel des Fensters, das erfasst werden soll.
        output_folder (str): Der Ordner zum Speichern des Screenshots.
    """
    print(f"Suche nach Fenster mit dem Titel: '{window_title}'...")

    # Versuche, das Fenster zu finden
    target_windows = gw.getWindowsWithTitle(window_title)
    if not target_windows:
        print(f"Fehler: Kein Fenster mit dem Titel '{window_title}' gefunden.")
        print("Bitte überprüfen Sie, ob das Spiel läuft und der Fenstertitel korrekt ist.")
        return

    window = target_windows[0]
    print("Fenster gefunden!")

    # Bringe das Fenster in den Vordergrund (optional, aber empfohlen)
    try:
        if window.isMinimized:
            window.restore()
        window.activate()
        time.sleep(0.5) # Kurze Pause, damit das Fenster Zeit hat zu reagieren
    except Exception as e:
        print(f"Konnte das Fenster nicht aktivieren: {e}")


    # Erstelle den Ausgabeordner, falls er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Ordner '{output_folder}' wurde erstellt.")

    # Hole die Dimensionen des Fensters
    left, top, width, height = window.left, window.top, window.width, window.height
    
    # Definiere die Bounding Box für den Screenshot
    bbox = (left, top, left + width, top + height)

    # Erstelle den Screenshot
    try:
        screenshot = ImageGrab.grab(bbox=bbox, all_screens=True)
        
        # Erzeuge einen eindeutigen Dateinamen mit Zeitstempel
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(output_folder, f"win_screenshot_{timestamp}.png")

        # Speichere die Datei
        screenshot.save(file_path)
        print(f"Screenshot erfolgreich gespeichert unter: {file_path}")

    except Exception as e:
        print(f"Fehler beim Erstellen des Screenshots: {e}")


if __name__ == "__main__":
    create_screenshot_from_window(WINDOW_TITLE, OUTPUT_FOLDER)